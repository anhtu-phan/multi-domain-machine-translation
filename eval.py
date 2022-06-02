import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from model import load_model
from constants import MODEL_TYPE, CONFIG


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, tokenizer=None, bpe=None):
    model.eval()
    if isinstance(sentence, str) and bpe is not None:
        sentence = bpe.process_line(sentence)
    if isinstance(sentence, str) and tokenizer is not None:
        sentence = tokenizer(sentence)

    tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # print(src_tensor)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        if len(data_dir) > 1:
            enc_src, _ = model.encoder(src_tensor, src_mask)
        else:
            enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            if len(data_dir) > 1 and CONFIG["MODEL_TYPE"] == MODEL_TYPE[1]:
                output, attention, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            else:
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
            # print(f"\nSkip long sentence {'-'*10}{'>'*10}")
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
    parser.add_argument("--data_dir", nargs='+', default=[])
    parser.add_argument("--test_data_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=int)

    args = parser.parse_args()

    data_dir = args.data_dir
    test_data_dir = args.test_data_dir
    saved_model_path = args.model_path
    model_type = args.model_type

    CONFIG["MODEL_TYPE"] = MODEL_TYPE[model_type]
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model, _, _, test_data, src, trg, _ = load_model(CONFIG, data_dir, test_data_dir, _device)
    checkpoint = torch.load(saved_model_path, map_location=torch.device(_device))
    _model.load_state_dict(checkpoint['state_dict'])
    bleu_score = calculate_bleu(test_data, src, trg, _model, _device)
    print(f"\n\n{'-'*10}BLEU_SCORE: {bleu_score:.2f}{'-'*10}")
