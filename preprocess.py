import os
import sys
import codecs
from learn_bpe import learn_bpe
from apply_bpe import BPE
from mosestokenizer import *

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


def build_bpe_data(data_dir, test_data_dir):
    # Build up the code from training files if not exist
    codes = f"{data_dir}/codes_bpe.txt"
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

    return os.path.join(data_dir, enc_train_files_prefix), os.path.join(data_dir, enc_val_files_prefix), os.path.join(data_dir, enc_test_files_prefix),
