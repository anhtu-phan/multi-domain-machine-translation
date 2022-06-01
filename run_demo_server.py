import os
import torch
from flask import Flask, request, render_template
import argparse
from apply_bpe import BPE
from learn_bpe import learn_bpe
import preprocess
import sys
import codecs
from mosestokenizer import *
from torchtext.legacy.data import LabelField, Field, TabularDataset
from model import load_model
from eval import translate_sentence

app = Flask(__name__)
tokenize_src = MosesTokenizer('en')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    input_sentence = request.form['input_sentence'].strip()
    result = []
    r, _ = translate_sentence(input_sentence, SRC, TRG, _model, device, 100, tokenize_src, bpe)
    result.append({"model_type": "News+TED", "result": r})
    r, _ = translate_sentence(input_sentence, SRC, TRG, _model_edc, device, 100, tokenize_src, bpe)
    result.append({"model_type": "E/DC", "result": r})
    r, _ = translate_sentence(input_sentence, SRC, TRG, _model_edc_with_int, 100, tokenize_src, bpe)
    result.append({"model_type": "E/DC with init", "result": r})
    r, _ = translate_sentence(input_sentence, SRC, TRG, _model_encoder, device, 100, tokenize_src, bpe)
    result.append({"model_type": "Encoder", "result": r})
    return render_template('index.html', result=result, input_sentence=input_sentence)


if __name__ == '__main__':
    CONFIG = {
        "LEARNING_RATE": 1e-7,
        "BATCH_SIZE": 32,
        "HID_DIM": 512,
        "ENC_LAYERS": 6,
        "DEC_LAYERS": 6,
        "ENC_HEADS": 4,
        "DEC_HEADS": 4,
        "ENC_PF_DIM": 1024,
        "DEC_PF_DIM": 1024,
        "ENC_DROPOUT": 0.2,
        "DEC_DROPOUT": 0.2,
        "N_EPOCHS": 1000000,
        "DOMAIN_EPS": 0.05,
        "CLIP": 1,
        "MODEL_TYPE": "edc"
    }

    parser = argparse.ArgumentParser(description="Transformer text recognition demo")
    parser.add_argument("--port", default=9595)
    parser.add_argument("--data_dir")

    args = parser.parse_args()
    port = args.port
    data_dir = args.data_dir

    train_src_path = f"{data_dir}/train.en"
    train_trg_path = f"{data_dir}/train.de"
    codes = f"{data_dir}/codes_bpe.txt"

    if not os.path.isfile(codes):
        sys.stderr.write(f"Collect codes from training data and save to {codes}.\n")
        learn_bpe([train_src_path, train_trg_path], codes, 32000, 6, tokenizer=[tokenize_src, tokenize_trg])
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(codes, encoding='utf-8') as codes:
        bpe = BPE(codes)

    tokenize_src = MosesTokenizer('en')
    tokenize_trg = MosesTokenizer("de")
    SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
                batch_first=True)
    TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
                batch_first=True)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, _, test_data = preprocess.read_data(SRC, TRG, data_folder=data_dir,
                                                    test_data_folder=data_dir,
                                                    use_bpe=True, max_length=100)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRC_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    sys.stderr.write(f"Load Direct training model ...\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = load_model(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRC_PAD_IDX, CONFIG, 1, device)
    checkpoint = torch.load("./checkpoints/model_de_en/model.pt", map_location=torch.device(_device))
    _model.load_state_dict(checkpoint['state_dict'])

    sys.stderr.write(f"Load EDC model ...\n")

    fields = [('src', SRC), ('trg', TRG), ('domain', LabelField())]

    train_data, valid_data, test_data = TabularDataset.splits(path="./datasets/de-en/news", train='train_combined.tsv',
                                                              test='test_combined.tsv',
                                                              validation='valid_combined.tsv',
                                                              format='tsv',
                                                              fields=fields, skip_header=True)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRC_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    _model_edc = load_model(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRC_PAD_IDX, CONFIG, 2, device)
    checkpoint = torch.load("./checkpoints/model_de_en/model_mutil.pt", map_location=torch.device(_device))
    _model_edc.load_state_dict(checkpoint['state_dict'])

    sys.stderr.write(f"Load EDC with init model ...\n")

    _model_edc_with_int = load_model(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRC_PAD_IDX, CONFIG, 2, device)
    checkpoint = torch.load("./checkpoints/model_de_en/model_mutil_with_init.pt", map_location=torch.device(_device))
    _model_edc_with_int.load_state_dict(checkpoint['state_dict'])

    sys.stderr.write(f"Load Encoder model ...\n")

    CONFIG["MODEL_TYPE"] = 'encoder'
    _model_encoder = load_model(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRC_PAD_IDX, CONFIG, 2, device)
    checkpoint = torch.load("./checkpoints/model_de_en/encoder_mutil_with_init.pt", map_location=torch.device(_device))
    _model_encoder.load_state_dict(checkpoint['state_dict'])

    app.run('0.0.0.0', port=port)
