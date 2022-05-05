import torch
import torch.nn as nn
from torchtext.legacy.data import BucketIterator
import wandb
from tqdm import tqdm
import os

from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq
from transformer_pytorch.optim import SchedulerOptim
from transformer_pytorch.loss import cal_performance
import preprocess


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, debugging=False):
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


def evaluate(model, iterator, debugging=False):
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

(SRC, TRG), train_data, valid_data, test_data = preprocess.main(use_bpe=True)

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

    train_loss, train_loss_per_word, train_acc = train(model=_model, iterator=train_iterator, optimizer=_optimizer)
    valid_loss, valid_loss_per_word, val_acc = evaluate(model=_model, iterator=valid_iterator)

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
