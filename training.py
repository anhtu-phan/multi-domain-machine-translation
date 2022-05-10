import torch
import torch.nn as nn
from torchtext.legacy.data import BucketIterator
import wandb
from tqdm import tqdm
import os
import argparse
import pandas as pd
from mosestokenizer import *
from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import Field, TabularDataset
from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq
from transformer_pytorch.domain_mixing_transformer import Encoder as DomainEncoder, Decoder as DomainDecoder, Seq2Seq as DomainSeq2Seq
from transformer_pytorch.optim import SchedulerOptim
from transformer_pytorch.loss import cal_performance
import preprocess
import build_dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, TRC_PAD_IDX, debugging=False):
    model.train()
    epoch_loss, epoch_word_total, epoch_n_word_correct = 0, 0, 0
    for i, batch in enumerate(iterator):
        if debugging and i == 2:
            break

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss, n_correct, n_word = cal_performance(output, trg, TRC_PAD_IDX, True, 0.1)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_word_total += n_word
        epoch_n_word_correct += n_correct

    loss_per_word = epoch_loss/epoch_word_total
    acc = epoch_n_word_correct/epoch_word_total

    return epoch_loss / len(iterator), loss_per_word, acc


def evaluate(model, iterator, TRC_PAD_IDX, debugging=False):
    model.eval()
    epoch_loss, epoch_word_total, epoch_n_word_correct = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if debugging and i == 2:
                break

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss, n_correct, n_word = cal_performance(output, trg, TRC_PAD_IDX, False, 0.1)
            epoch_loss += loss.item()
            epoch_word_total += n_word
            epoch_n_word_correct += n_correct

    return epoch_loss / len(iterator), epoch_loss/epoch_word_total, epoch_n_word_correct/epoch_word_total


def read_data(SRC, TRG, data_folder, test_data_folder, use_bpe=False, max_length=100):
    train_data_path, val_data_path, test_data_path = preprocess.build_bpe_data(data_folder, test_data_folder)

    if not use_bpe:
        fields = [('src', SRC), ('trg', TRG)]

        train_data, valid_data, test_data = TabularDataset.splits(path=data_dir, train='train.tsv',
                                                                  test='test.tsv', validation='valid.tsv', format='tsv',
                                                                  fields=fields, skip_header=True)
        return (SRC, TRG), train_data, valid_data, test_data

    fields = (SRC, TRG)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= max_length and len(vars(x)['trg']) <= max_length

    train = TranslationDataset(
        fields=fields,
        path=train_data_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    val = TranslationDataset(
        fields=fields,
        path=val_data_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    test = TranslationDataset(
        fields=fields,
        path=test_data_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    return train, val, test


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenize_src = MosesTokenizer('en')
    tokenize_trg = MosesTokenizer("de")
    SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True, batch_first=True)

    if len(data_dir) > 1:
        DOMAIN = Field(is_target=True)
        for i, d in enumerate(data_dir):
            train_type, valid_type, test_type = preprocess.build_bpe_data(d, d)
            if i == 0:
                train_data = build_dataset.build_data(d, train_type, 'src', 'trg', i)
                valid_data = build_dataset.build_data(d, valid_type, 'src', 'trg', i)
                test_data = build_dataset.build_data(d, test_type, 'src', 'trg', i)
            else:
                tr = build_dataset.build_data(d, train_type, 'src', 'trg', i)
                v = build_dataset.build_data(d, valid_type, 'src', 'trg', i)
                te = build_dataset.build_data(d, test_type, 'src', 'trg', i)
                train_data = pd.concat([train_data, tr])
                valid_data = pd.concat([valid_data, v])
                test_data = pd.concat([test_data, te])
        train_data.to_csv(f"{data_dir[0]}/train_combined.tsv", sep="\t", index=False)
        valid_data.to_csv(f"{data_dir[0]}/valid_combined.tsv", sep="\t", index=False)
        test_data.to_csv(f"{data_dir[0]}/test_combined.tsv", sep="\t", index=False)

        fields = [('src', SRC), ('trg', TRG), ('domain', DOMAIN)]

        train_data, valid_data, test_data = TabularDataset.splits(path=data_dir[0], train='train_combined.tsv',
                                                                  test='test_combined.tsv', validation='valid_combined.tsv', format='tsv',
                                                                  fields=fields, skip_header=True)
    else:
        train_data, valid_data, test_data = read_data(SRC, TRG, data_folder=data_dir, test_data_folder=test_data_dir,
                                                      use_bpe=True, max_length=100)

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                          sort_key=lambda x: len(x.src),
                                                                          sort_within_batch=False,
                                                                          batch_size=CONFIG["BATCH_SIZE"], device=device)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    enc = Encoder(INPUT_DIM, CONFIG['HID_DIM'], CONFIG['ENC_LAYERS'], CONFIG['ENC_HEADS'], CONFIG['ENC_PF_DIM'],
                  CONFIG['ENC_DROPOUT'], device)
    dec = Decoder(OUTPUT_DIM, CONFIG['HID_DIM'], CONFIG['DEC_LAYERS'], CONFIG['DEC_HEADS'], CONFIG['DEC_PF_DIM'],
                  CONFIG['DEC_DROPOUT'], device)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRC_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]


    _model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRC_PAD_IDX, device).to(device)

    model_name = 'model.pt'
    saved_model_dir = './checkpoints/model_de_en/'
    saved_model_path = saved_model_dir+model_name
    best_valid_loss = float('inf')
    saved_epoch = 0

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    if os.path.exists(saved_model_path):
        print(f"Load saved model {'.'*10}")
        last_checkpoint = torch.load(saved_model_path, map_location=torch.device(device))
        best_valid_loss = last_checkpoint['best_valid_loss']
        saved_epoch = last_checkpoint['epoch']
        _model.load_state_dict(last_checkpoint['state_dict'])
        CONFIG['LEARNING_RATE'] = last_checkpoint['lr']
    else:
        _model.apply(initialize_weights)

    _optimizer = SchedulerOptim(torch.optim.Adam(_model.parameters(), lr=CONFIG['LEARNING_RATE'], betas=(0.9, 0.98),
                                                 weight_decay=0.0001), 1, CONFIG['HID_DIM'], 4000, 5e-4, saved_epoch)

    wandb.init(name="training-transformer-en2de", project="multi-domain-machine-translation", config=CONFIG, resume=True)
    wandb.watch(_model, log='all')

    for epoch in tqdm(range(saved_epoch, CONFIG['N_EPOCHS'])):
        logs = dict()

        train_lr = _optimizer.optimizer.param_groups[0]['lr']
        logs['train_lr'] = train_lr

        train_loss, train_loss_per_word, train_acc = train(model=_model, iterator=train_iterator, optimizer=_optimizer, TRC_PAD_IDX=TRC_PAD_IDX)
        valid_loss, valid_loss_per_word, val_acc = evaluate(model=_model, iterator=valid_iterator, TRC_PAD_IDX=TRC_PAD_IDX)

        logs['train_loss'] = train_loss
        logs['train_loss_per_word'] = train_loss_per_word
        logs['train_acc'] = train_acc
        logs['valid_loss'] = valid_loss
        logs['valid_loss_per_word'] = valid_loss_per_word
        logs['val_acc'] = val_acc

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint = {
                'epoch': epoch+1,
                'state_dict': _model.state_dict(),
                'best_valid_loss': best_valid_loss,
                'lr': train_lr
            }
            torch.save(checkpoint, saved_model_path)

        wandb.log(logs, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mutil domain machine translation evaluation")
    parser.add_argument("--data_dir", nargs='+', default=[])
    parser.add_argument("--test_data_dir", type=str)

    args = parser.parse_args()

    data_dir = args.data_dir
    test_data_dir = args.test_data_dir

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
    main()
