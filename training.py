import torch
import torch.nn as nn
import spacy
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import wandb
from tqdm import tqdm

from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', fix_length=100, lower=True, batch_first=True)

fields = [('source', SRC), ('target', TRG)]

train_data, valid_data, test_data = TabularDataset.splits(path='./datasets/de-en/mixed', train='train.tsv', test='test.tsv', validation='valid.tsv', format='tsv', fields=fields, skip_header=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    "LEARNING_RATE": 0.0005,
    "BATCH_SIZE": 16,
    "HID_DIM": 512,
    "ENC_LAYERS": 6,
    "DEC_LAYERS": 6,
    "ENC_HEADS": 4,
    "DEC_HEADS": 4,
    "ENC_PF_DIM": 1024,
    "DEC_PF_DIM": 1024,
    "ENC_DROPOUT": 0.1,
    "DEC_DROPOUT": 0.1,
    "N_EPOCHS": 10,
    "CLIP": 1,
}

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=CONFIG["BATCH_SIZE"], device=device)

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

enc = Encoder(INPUT_DIM, CONFIG['HID_DIM'], CONFIG['ENC_LAYERS'], CONFIG['ENC_HEADS'], CONFIG['ENC_PF_DIM'], CONFIG['ENC_DROPOUT'], device)
dec = Decoder(OUTPUT_DIM, CONFIG['HID_DIM'], CONFIG['DEC_LAYERS'], CONFIG['DEC_HEADS'], CONFIG['DEC_PF_DIM'], CONFIG['DEC_DROPOUT'], device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRC_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRC_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model)} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

criterion = nn.CrossEntropyLoss(ignore_index=TRC_PAD_IDX)

wandb.init(name="training-transformer-en2de", project="multi-domain-machine-translation", config=CONFIG, resume=True)
wandb.watch(model, log='all')


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.source
        trg = batch.target

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


saved_model_path = './checkpoints/model_de_en/'
best_valid_loss = float('inf')
saved_epoch = 0

for epoch in tqdm(range(saved_epoch, CONFIG['N_EPOCHS'])):
    logs = dict()

    train_loss = train(model, train_iterator, optimizer, criterion, CONFIG['CLIP'])
    valid_loss = evaluate(model, valid_iterator, criterion)
    logs['train_loss'] = train_loss
    logs['valid_loss'] = valid_loss

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'{saved_model_path}/model.pt')

    wandb.log(logs, step=epoch)
