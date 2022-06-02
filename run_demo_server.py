import os
import torch
from flask import Flask, request, render_template
import argparse
from apply_bpe import BPE
from learn_bpe import learn_bpe
import sys
import codecs
from mosestokenizer import *
from model import load_model
from eval import translate_sentence
from constants import MODEL_TYPE, CONFIG

app = Flask(__name__)
tokenize_src = MosesTokenizer('en')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    input_sentence = request.form['input_sentence'].strip()
    result = []
    r, _ = translate_sentence(input_sentence, model_src, model_trg, _model, device, 100, tokenize_src, model_bpe)
    result.append({"model_type": "News+TED", "result": r})
    r, _ = translate_sentence(input_sentence, model_edc_src, model_edc_trg, _model_edc, device, 100, tokenize_src, model_edc_bpe)
    result.append({"model_type": "E/DC", "result": r})
    r, _ = translate_sentence(input_sentence, model_edc_with_int_src, model_edc_with_int_trg, _model_edc_with_int, 100, tokenize_src, model_edc_with_int_bpe)
    result.append({"model_type": "E/DC with init", "result": r})
    r, _ = translate_sentence(input_sentence, model_encoder_src, model_encoder_trg, _model_encoder, device, 100, tokenize_src, model_encoder_bpe)
    result.append({"model_type": "Encoder", "result": r})
    return render_template('index.html', result=result, input_sentence=input_sentence)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Transformer text recognition demo")
    parser.add_argument("--port", default=9595)
    parser.add_argument("--data_dir_mixed", type=str)
    parser.add_argument("--data_dir_domain", nargs='+', default=[])

    args = parser.parse_args()
    port = args.port
    data_dir_mixed = args.data_dir_mixed
    data_dir_domain = args.data_dir_domain

    sys.stderr.write(f"Load Direct training model ...\n")
    _model, _, _, _, model_src, model_trg, model_bpe = load_model(CONFIG, data_dir_mixed, data_dir_mixed, device)
    checkpoint = torch.load("./checkpoints/model_de_en/model.pt", map_location=torch.device(device))
    _model.load_state_dict(checkpoint['state_dict'])

    sys.stderr.write(f"Load EDC model ...\n")

    _model_edc, _, _, _, model_edc_src, model_edc_trg, model_edc_bpe = load_model(CONFIG, data_dir_domain, data_dir_domain[0], device)
    checkpoint = torch.load("./checkpoints/model_de_en/model_mutil.pt", map_location=torch.device(device))
    _model_edc.load_state_dict(checkpoint['state_dict'])

    sys.stderr.write(f"Load EDC with init model ...\n")

    _model_edc_with_int, _, _, _, model_edc_with_int_src, model_edc_with_int_trg, model_edc_with_int_bpe = \
        load_model(CONFIG, data_dir_domain, data_dir_domain[0], device)
    checkpoint = torch.load("./checkpoints/model_de_en/model_mutil_with_init.pt", map_location=torch.device(device))
    _model_edc_with_int.load_state_dict(checkpoint['state_dict'])

    sys.stderr.write(f"Load Encoder model ...\n")

    CONFIG["MODEL_TYPE"] = MODEL_TYPE[2]
    _model_encoder, _, _, _, model_encoder_src, model_encoder_trg, model_encoder_bpe \
        = load_model(CONFIG, data_dir_domain, data_dir_domain[0], device)
    checkpoint = torch.load("./checkpoints/model_de_en/encoder_mutil_with_init.pt", map_location=torch.device(device))
    _model_encoder.load_state_dict(checkpoint['state_dict'])

    app.run('0.0.0.0', port=port)
