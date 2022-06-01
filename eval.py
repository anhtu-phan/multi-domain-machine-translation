import argparse

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from mosestokenizer import *
from torchtext.legacy.data import Field

from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq
import preprocess


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, tokenizer=None):
    model.eval()

    if isinstance(sentence, str) and tokenizer is not None:
        sentence = tokenizer(sentence)

    tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # print(src_tensor)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'], rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
    lst_trg = []
    lst_pred_trg = []
    nb_skip = 0
    for i in tqdm(range(len(data))):
        src = vars(data[i])['src']
        trg = vars(data[i])['trg']
        if len(src) > 98:
            print(f"\nSkip long sentence {'-'*10}{'>'*10}")
            nb_skip += 1
            continue
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        # print(f"\ntrg: {trg}\npred:{pred_trg}")

        pred_trg = pred_trg[:-1]
        lst_pred_trg.append(pred_trg)
        lst_trg.append([trg])

    print(f"\n{'-' * 10}nb_skip = {nb_skip}{'-' * 10}")
    return bleu_score(lst_pred_trg, lst_trg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mutil domain machine translation evaluation")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--test_data_dir", type=str)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()

    data_dir = args.data_dir
    test_data_dir = args.test_data_dir
    saved_model_path = args.model_path

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
        "CLIP": 1
    }
    tokenize_src = MosesTokenizer('en')
    tokenize_trg = MosesTokenizer("de")
    SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
                batch_first=True)
    TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
                batch_first=True)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, _, test_data = preprocess.read_data(SRC, TRG, data_folder=data_dir,
                                                    test_data_folder=test_data_dir,
                                                    use_bpe=True, max_length=100)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    enc = Encoder(INPUT_DIM, CONFIG['HID_DIM'], CONFIG['ENC_LAYERS'], CONFIG['ENC_HEADS'], CONFIG['ENC_PF_DIM'],
                  CONFIG['ENC_DROPOUT'], _device)
    dec = Decoder(OUTPUT_DIM, CONFIG['HID_DIM'], CONFIG['DEC_LAYERS'], CONFIG['DEC_HEADS'], CONFIG['DEC_PF_DIM'],
                  CONFIG['DEC_DROPOUT'], _device)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRC_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    _model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRC_PAD_IDX, _device).to(_device)
    checkpoint = torch.load(saved_model_path, map_location=torch.device(_device))
    _model.load_state_dict(checkpoint['state_dict'])
    bleu_score = calculate_bleu(test_data, SRC, TRG, _model, _device)
    print(f"\n\n{'-'*10}BLEU_SCORE: {bleu_score:.2f}{'-'*10}")
