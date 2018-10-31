import glob
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboard import TensorBoard
from torchtext import data, datasets

from model import NLI


# training settings
parser = ArgumentParser(description='SNLI')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--encoder', type=str, default='rnn')
# TODO: link size of d_embed to the pre-trained word-embedding
parser.add_argument('--d_embed', type=int, default=300)
parser.add_argument('--word_vectors', type=str, default='fasttext.en.300d')
parser.add_argument('--d_hidden', type=int, default=300)
parser.add_argument('--d_fc', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--dp_ratio', type=float, default=0.2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_file', type=str, default='snli_train.tsv')
parser.add_argument('--val_file', type=str, default='snli_val.tsv')
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--dev_every', type=int, default=1000)
parser.add_argument('--experiment', type=str, default='test')
params = parser.parse_args()


# gpu business
if torch.cuda.is_available():
    torch.cuda.set_device(params.gpu)
    device = torch.device('cuda:{}'.format(params.gpu))
else:
    device = torch.device('cpu')

# tensoboard logging
model_dir = f"runs/{params.experiment}/{time.asctime(time.localtime())}/"
tb = TensorBoard(model_dir)


# define text felids
# TODO: Other tokenizers?
inputs = data.Field(lower=True, tokenize='spacy')
answers = data.Field(sequential=False, unk_token=None)


train, valid = data.TabularDataset.splits(
    path="data",
    train=params.train_file, validation=params.val_file,
    format='tsv',
    skip_header=True,
    fields=[("sentence1", inputs), ("sentence2", inputs), ("label", answers)])


# build vocabulary and load per-trained embeddings
# TODO: Too slow for big dataset. Use n-workers.
inputs.build_vocab(train, valid, vectors=params.word_vectors)
answers.build_vocab(train)

# equivalent of dataloader in torchtext
train_iter, valid_iter = data.BucketIterator.splits(
            (train, valid), 
            batch_size=params.batch_size, 
            sort_key=lambda x: len(x.sentence1),
            device=device)

params.n_embed = len(inputs.vocab)
params.d_out = len(answers.vocab)
# double the number of cells for bidirectional networks
params.n_cells = params.n_layers * 2


model = NLI(params)
# hack to tie pre-trained vectors to Embedding class
# TODO: find a cleaner way
if params.word_vectors:
    model.embed.weight.data.copy_(inputs.vocab.vectors)
model.to(device)

print("Total number of parameters: {}.".format(sum(p.numel()
                                                   for p in model.parameters() if p.requires_grad)))

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=params.lr)


# global step for tensorboard logging
step = 0
start = time.time()

def training():
    # logging business
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss       Accuracy'
    print(header)

    global step
    for epoch in range(params.epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):
            step += 1

            model.train()
            opt.zero_grad()
            answer = model(batch.sentence1, batch.sentence2)
            loss = criterion(answer, batch.label)
            loss.backward()
            opt.step()

            # evaluate performance on validation set periodically
            if step % params.dev_every == 0:
                validation()

            if step % params.log_every == 0:
                # calculate accuracy
                n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                n_total += batch.batch_size
                accuracy = 100. * n_correct/n_total

                # print progress message
                log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f}'.split(','))
                print(log_template.format(time.time()-start, epoch, step, 1+batch_idx, len(train_iter),
                        100. * (1+batch_idx) / len(train_iter), loss.item(), accuracy))
                
                if tb is not None:
                    tb.scalar_summary("train/loss", loss.item(), step)
                    tb.scalar_summary("train/accuracy", accuracy, step)



def validation():
    global step
    # switch model to evaluation mode
    model.eval()
    valid_iter.init_epoch()

    # calculate accuracy on validation set
    n_valid_correct, valid_loss = 0, 0
    with torch.no_grad():
        for _, valid_batch in enumerate(valid_iter):
            answer = model(valid_batch.sentence1, valid_batch.sentence2)
            n_valid_correct += (torch.max(answer, 1)[1].view(valid_batch.label.size()) == valid_batch.label).sum().item()
            valid_loss = criterion(answer, valid_batch.label)
    valid_acc = 100. * n_valid_correct / len(valid)

    valid_log_template = 'Validation Loss: {:>8.6f}, Accuracy: {:12.4f}'
    print(valid_log_template.format(valid_loss.item(), valid_acc))

    if tb is not None:
        tb.scalar_summary("validation/loss", valid_loss.item(), step)
        tb.scalar_summary("validation/accuracy", valid_acc, step)




def main():
    print(params)
    training()

if __name__ == '__main__':
    main()

