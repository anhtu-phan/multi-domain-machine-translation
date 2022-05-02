import torch
import torch.nn as nn
from torchtext.legacy.data import BucketIterator
import wandb
from tqdm import tqdm
import os

from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq
from transformer_pytorch.optim import SchedulerOptim
import preprocess


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


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

(SRC, TRG), train_data, valid_data, test_data = preprocess.main(use_bpe=False)

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

saved_model_path = './checkpoints/model_de_en/'
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

best_valid_loss = float('inf')
saved_epoch = 0

_model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRC_PAD_IDX, device).to(device)

_model.apply(initialize_weights)

_optimizer = SchedulerOptim(torch.optim.Adam(_model.parameters(), lr=CONFIG['LEARNING_RATE'], betas=(0.9, 0.98),
                                             weight_decay=0.0001), 1, CONFIG['HID_DIM'], 4000, 5e-4, saved_epoch)

_criterion = nn.CrossEntropyLoss(ignore_index=TRC_PAD_IDX, label_smoothing=0.1)

wandb.init(name="training-transformer-en2de", project="multi-domain-machine-translation", config=CONFIG, resume=False)
wandb.watch(_model, log='all')

for epoch in tqdm(range(saved_epoch, CONFIG['N_EPOCHS'])):
    logs = dict()

    train_loss = train(model=_model, iterator=train_iterator, optimizer=_optimizer, criterion=_criterion, clip=CONFIG['CLIP'])
    valid_loss = evaluate(model=_model, iterator=valid_iterator, criterion=_criterion)
    train_lr = _optimizer.optimizer.param_groups[0]['lr']
    logs['train_loss'] = train_loss
    logs['valid_loss'] = valid_loss
    logs['train_lr'] = train_lr

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(_model.state_dict(), f'{saved_model_path}/model.pt')

    wandb.log(logs, step=epoch)