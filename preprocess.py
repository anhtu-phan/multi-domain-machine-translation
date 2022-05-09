import os
import sys
import codecs
import argparse
from mosestokenizer import *
from torchtext.legacy.data import Field, TabularDataset
from torchtext.legacy.datasets import TranslationDataset
from learn_bpe import learn_bpe
from apply_bpe import BPE

tokenize_src = MosesTokenizer('en')
tokenize_trg = MosesTokenizer("de")


def encode_file(bpe, in_file, out_file):
    sys.stderr.write(f"Read raw content from {in_file} and \n"\
            f"Write encoded content to {out_file}\n")
    
    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))


def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix):
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    # if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
    #     sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file)
    encode_file(bpe, trg_in_file, trg_out_file)
    return src_out_file, trg_out_file


def main(data_dir, test_data_dir, max_length, use_bpe=False):
    if use_bpe:
        return build_bpe_data(data_dir, test_data_dir, max_length)
    else:
        SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
                    batch_first=True)
        TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True,
                    batch_first=True)

        fields = [('src', SRC), ('trg', TRG)]

        train_data, valid_data, test_data = TabularDataset.splits(path=data_dir, train='train.tsv',
                                                                  test='test.tsv', validation='valid.tsv', format='tsv',
                                                                  fields=fields, skip_header=True)
        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)

        return (SRC, TRG), train_data, valid_data, test_data


def build_bpe_data(data_dir, test_data_dir, max_length):
    # Build up the code from training files if not exist
    codes = "./datasets/de-en/mixed/codes_bpe.txt"
    train_src_path = f"{data_dir}/train.en"
    train_trg_path = f"{data_dir}/train.de"
    val_src_path = f"{data_dir}/valid.en"
    val_trg_path = f"{data_dir}/valid.de"
    test_src_path = f"{test_data_dir}/test.en"
    test_trg_path = f"{test_data_dir}/test.de"
    enc_train_files_prefix = 'bpe-train'
    enc_val_files_prefix = 'bpe-val'
    enc_test_files_prefix = 'bpe-test'

    if not os.path.isfile(codes):
        sys.stderr.write(f"Collect codes from training data and save to {codes}.\n")
        learn_bpe([train_src_path, train_trg_path], codes, 32000, 6, tokenizer=[tokenize_src, tokenize_trg])
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(codes, encoding='utf-8') as codes:
        bpe = BPE(codes)

    sys.stderr.write(f"Encoding ...\n")

    encode_files(bpe, train_src_path, train_trg_path, data_dir, enc_train_files_prefix)
    encode_files(bpe, val_src_path, val_trg_path, data_dir, enc_val_files_prefix)
    encode_files(bpe, test_src_path, test_trg_path, data_dir, enc_test_files_prefix)
    sys.stderr.write(f"Done.\n")

    SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>',
                fix_length=max_length, lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>',
                fix_length=max_length, lower=True, batch_first=True)

    fields = (SRC, TRG)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= max_length and len(vars(x)['trg']) <= max_length

    train = TranslationDataset(
        fields=fields,
        path=os.path.join(data_dir, enc_train_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    val = TranslationDataset(
        fields=fields,
        path=os.path.join(data_dir, enc_val_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    test = TranslationDataset(
        fields=fields,
        path=os.path.join(data_dir, enc_test_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    SRC.build_vocab(train, min_freq=2)
    TRG.build_vocab(train, min_freq=2)

    return fields, train, val, test
