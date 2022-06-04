import torch
import torch.nn as nn
from torchtext.legacy.data import BucketIterator
import wandb
from tqdm import tqdm
import os
import argparse
from transformer_pytorch.optim import SchedulerOptim
from transformer_pytorch.loss import cal_performance, cal_domain_loss
from constants import MODEL_TYPE, CONFIG
from model import load_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, trg_pad_idx, mutil_domain=False, debugging=False):
    model.train()
    epoch_loss, epoch_loss_domain, epoch_word_total, epoch_n_word_correct = 0, 0, 0, 0
    for i, batch in enumerate(tqdm(iterator)):
        if debugging and i == 2:
            break

        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        if mutil_domain:
            output, _, domain_prob = model(src, trg[:, :-1])
        else:
            output, _ = model(src, trg[:, :-1])

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss, n_correct, n_word = cal_performance(output, trg, trg_pad_idx, True, 0.1)
        if mutil_domain:
            domain = batch.domain
            l_mix = cal_domain_loss(domain, domain_prob)
            loss += l_mix
            epoch_loss_domain += l_mix.item()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_word_total += n_word
        epoch_n_word_correct += n_correct

    loss_per_word = epoch_loss/epoch_word_total
    acc = epoch_n_word_correct/epoch_word_total

    return epoch_loss / len(iterator), loss_per_word, acc, epoch_loss_domain / len(iterator)


def evaluate(model, iterator, trg_pad_idx, mutil_domain=False, debugging=False):
    model.eval()
    epoch_loss, epoch_word_total, epoch_n_word_correct = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            if debugging and i == 2:
                break

            src = batch.src
            trg = batch.trg
            if mutil_domain:
                output, _, _ = model(src, trg[:, :-1])
            else:
                output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss, n_correct, n_word = cal_performance(output, trg, trg_pad_idx, False, 0.1)
            epoch_loss += loss.item()
            epoch_word_total += n_word
            epoch_n_word_correct += n_correct

    return epoch_loss / len(iterator), epoch_loss/epoch_word_total, epoch_n_word_correct/epoch_word_total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _model, train_data, valid_data, test_data, _, trg, _ = load_model(CONFIG, data_dir, data_dir[0], device)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                          sort_key=lambda x: len(x.src),
                                                                          sort_within_batch=False,
                                                                          batch_size=CONFIG["BATCH_SIZE"],
                                                                          shuffle=True,
                                                                          device=device)

    print(f"{'-'*10}number of parameters = {count_parameters(_model)}{'-'*10}\n")
    model_name = f'{CONFIG["MODEL_TYPE"]}.pt'
    wandb_name = f'{CONFIG["MODEL_TYPE"]}'
    saved_model_dir = './checkpoints/model_de_en/'
    saved_model_path = saved_model_dir+model_name
    best_valid_loss = float('inf')
    saved_epoch = 0

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    if os.path.exists(saved_model_path):
        print(f"Load saved model {'.'*10}\n")
        last_checkpoint = torch.load(saved_model_path, map_location=torch.device(device))
        best_valid_loss = last_checkpoint['best_valid_loss']
        saved_epoch = last_checkpoint['epoch']
        _model.load_state_dict(last_checkpoint['state_dict'])
        CONFIG['LEARNING_RATE'] = last_checkpoint['lr']
        wandb.init(name=wandb_name, project="multi-domain-machine-translation", config=CONFIG,
                   resume=True)
    else:
        _model.apply(initialize_weights)
        wandb.init(name=wandb_name, project="multi-domain-machine-translation", config=CONFIG,
                   resume=False)

    _optimizer = SchedulerOptim(torch.optim.Adam(_model.parameters(), lr=CONFIG['LEARNING_RATE'], betas=(0.9, 0.98),
                                                 weight_decay=0.0001), 1, CONFIG['HID_DIM'], 4000, 5e-4, saved_epoch)

    wandb.watch(_model, log='all')

    for epoch in tqdm(range(saved_epoch, CONFIG['N_EPOCHS'])):
        logs = dict()

        train_lr = _optimizer.optimizer.param_groups[0]['lr']
        logs['train_lr'] = train_lr

        train_loss, train_loss_per_word, train_acc, train_domain_loss = train(model=_model, iterator=train_iterator, optimizer=_optimizer,
                                                           trg_pad_idx=trg.vocab.stoi[trg.pad_token], mutil_domain=(len(data_dir) > 1),)
        valid_loss, valid_loss_per_word, val_acc = evaluate(model=_model, iterator=valid_iterator,
                                                            trg_pad_idx=trg.vocab.stoi[trg.pad_token], mutil_domain=(len(data_dir) > 1))

        logs['train_loss'] = train_loss
        logs['train_loss_per_word'] = train_loss_per_word
        logs['train_acc'] = train_acc
        logs['train_domain_loss'] = train_domain_loss
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
    parser.add_argument("--model_type", type=int)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_type = args.model_type
    CONFIG['MODEL_TYPE'] = MODEL_TYPE[model_type]
    main()
