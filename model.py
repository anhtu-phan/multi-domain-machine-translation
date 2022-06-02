import os.path

import torch
import pandas as pd
from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq
from transformer_pytorch.domain_mixing_transformer import Encoder as DomainEncoder, Decoder as DomainDecoder, Seq2Seq as DomainSeq2Seq
from constants import MODEL_TYPE
from mosestokenizer import *
from torchtext.legacy.data import LabelField, Field, TabularDataset
import preprocess
import build_dataset

tokenize_src = MosesTokenizer('en')
tokenize_trg = MosesTokenizer("de")
src = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
            batch_first=True)
trg = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
            batch_first=True)


def load_data(data_dir, test_data_dir):
    if len(data_dir) > 1:
        domain = LabelField()
        for i, d in enumerate(data_dir):
            train_type, valid_type, test_type, bpe = preprocess.build_bpe_data(d, d)
            tr = build_dataset.build_data(train_type, 'src', 'trg', i)
            v = build_dataset.build_data(valid_type, 'src', 'trg', i)
            te = build_dataset.build_data(test_type, 'src', 'trg', i)
            if i == 0:
                train_data = tr
                valid_data = v
                test_data = te
            else:
                train_data = pd.concat([train_data, tr])
                valid_data = pd.concat([valid_data, v])
                test_data = pd.concat([test_data, te])
        train_data.to_csv(f"{data_dir[0]}/train_combined.tsv", sep="\t", index=False)
        valid_data.to_csv(f"{data_dir[0]}/valid_combined.tsv", sep="\t", index=False)
        test_data.to_csv(f"{data_dir[0]}/test_combined.tsv", sep="\t", index=False)

        fields = [('src', src), ('trg', trg), ('domain', domain)]

        train_data, valid_data, test_data = TabularDataset.splits(path=data_dir[0], train='train_combined.tsv',
                                                                  test='test_combined.tsv',
                                                                  validation='valid_combined.tsv',
                                                                  format='tsv',
                                                                  fields=fields, skip_header=True)
        domain.build_vocab(train_data)
    else:
        train_data, valid_data, test_data, bpe = preprocess.read_data(src, trg, data_folder=data_dir[0],
                                                                      test_data_folder=test_data_dir,
                                                                      use_bpe=True, max_length=100)

    src.build_vocab(train_data, min_freq=2)
    trg.build_vocab(train_data, min_freq=2)

    return train_data, valid_data, test_data, bpe


def load_model(config, data_dir, test_data_dir, device):
    train_data, valid_data, test_data, bpe = load_data(data_dir, test_data_dir)

    input_dim = len(src.vocab)
    output_dim = len(trg.vocab)
    src_pad_idx = src.vocab.stoi[src.pad_token]
    trg_pad_idx = trg.vocab.stoi[trg.pad_token]

    if len(data_dir) > 1:
        enc = DomainEncoder(input_dim, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'],
                            config['ENC_PF_DIM'], config['ENC_DROPOUT'], len(data_dir), config['DOMAIN_EPS'], device)
        if config['MODEL_TYPE'] == MODEL_TYPE[2]:
            print(f"{'-' * 10}Construct domain mixing ENCODER network{'-' * 10}")
            dec = Decoder(output_dim, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                          config['DEC_PF_DIM'], config['DEC_DROPOUT'], device)
            model = DomainSeq2Seq(enc, dec, src_pad_idx, trg_pad_idx, True, device).to(device)
        else:
            print(f"{'-' * 10}Construct domain mixing EDC network{'-' * 10}")
            dec = DomainDecoder(output_dim, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                                config['DEC_PF_DIM'], config['DEC_DROPOUT'], len(data_dir), config['DOMAIN_EPS'], device)
            model = DomainSeq2Seq(enc, dec, src_pad_idx, trg_pad_idx, False, device).to(device)
    else:
        print(f"{'-' * 10}Construct original network{'-' * 10}")
        enc = Encoder(input_dim, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device)
        dec = Decoder(output_dim, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'], config['DEC_PF_DIM'],
                      config['DEC_DROPOUT'], device)
        model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    return model, train_data, valid_data, test_data, src, trg, bpe
