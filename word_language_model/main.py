# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data
from data import rearrange
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--input_train', type=str, default='./data/input_train.out',
                    help='location of the input corpus')
parser.add_argument('--target_train', type=str, default='./data/target_train.out',
                    help='location of the target corpus')
parser.add_argument('--input_val', type=str, default='./data/input_val.out',
                    help='location of the input corpus')
parser.add_argument('--target_val', type=str, default='./data/target_val.out',
                    help='location of the target corpus')
parser.add_argument('--input_test', type=str, default='./data/input_test.out',
                    help='location of the input corpus')
parser.add_argument('--target_test', type=str, default='./data/target_test.out',
                    help='location of the target corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=5,
                    help='size of data')
# parser.add_argument('--emsize', type=int, default=200,
#                     help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# corpus = data.Corpus(args.data)
num = 1000000
# num = 7000000
#num = 100000
seq_len = 30
corpus = data.Corpus(args.input_train, args.target_train, args.input_val, args.target_val, args.input_test, args.target_test, num, seq_len)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(input_data, target_data, bsz):
    print("Batchifying data....")
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = input_data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    input_data = input_data.narrow(0, 0, nbatch * bsz)
    target_data = target_data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    input_data = input_data.view(input_data.shape[0] // bsz, bsz, input_data.shape[1], -1)
    target_data = target_data.view(bsz, -1).t().contiguous()
    # if args.cuda:
    #     input_data = input_data.cuda()
    #     target_data = target_data.cuda()
    return input_data, target_data

# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     if args.cuda:
#         data = data.cuda()
#     return data

eval_batch_size = 10
# train_data = batchify(corpus.train, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)
# test_data = batchify(corpus.test, eval_batch_size)

train_in, train_tar = batchify(corpus.train_in, corpus.train_tar, args.batch_size)
val_in, val_tar = batchify(corpus.val_in, corpus.val_tar, args.batch_size)
test_in, test_tar = batchify(corpus.test_in, corpus.test_tar, args.batch_size)

###############################################################################
# Build the model
###############################################################################

# ntokens = len(corpus.dictionary)
ntokens = 2
#model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
model = model.RNNModel(args.model, ntokens, seq_len * 5, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(input_data, target_data, i, evaluation=False):
    import pdb; pdb.set_trace()
    seq_len = min(args.bptt, len(input_data) - 1 - i)
    
    data = input_data[i:i+seq_len]
    target = target_data[i+i+seq_len]
    if args.cuda:
        data = data.cuda()
        target = target.cuda()

    data = Variable(data, volatile=evaluation)
    target = Variable(target)

    return data, target

def evaluate(input_data, target_data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = 2
    correct = 0
    total = 0
    hidden = model.init_hidden(args.batch_size)
    # hidden = model.init_hidden(eval_batch_size)
    # for i in range(0, data_source.size(0) - 1, args.bptt):
    for batch, i in enumerate(range(0, input_data.size(0), args.bptt)):
        data, targets = get_batch(input_data, target_data, i)
        data = data.squeeze(3)
        # data = Variable(rearrange(train_in[i]))
        # data = data.t().contiguous()
        # targets = Variable(targets[i])
        output, hidden = model(data, hidden)
        #targets = targets.view(targets.shape[0], 1)
        # output_flat = output.view(-1, ntokens)
        # total_loss += len(data) * criterion(output_flat, targets).data
        total_loss += len(data) * criterion(output.view(-1, ntokens), targets.long().view(-1)).data
        hidden = repackage_hidden(hidden)
    # return total_loss[0] / len(data_source)
        #import pdb; pdb.set_trace()
        pred = output.data.max(1, keepdim=True)[1]
        #pred = output.data.max(2, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
        total += output.shape[0] * output.shape[1]
    return total_loss[0] / len(input_data), correct/total


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = 2
    hidden = model.init_hidden(args.batch_size)
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    #     data, targets = get_batch(train_data, i)
    # print("Train data size...")
    # print(train_in.shape)
    # print(train_in.size(0))
    for batch, i in enumerate(range(0, train_in.size(0) - 1, args.bptt)):
        import pdb; pdb.set_trace()
        print("Starting trainig iteration {}".format(i))
        data, targets = get_batch(train_in, train_tar, i)
        # Rearrange data to be in the shape of seq_len x batch size x input size
        data = data.squeeze(3)
        #data = Variable(rearrange(train_in[i]))
        #data = data.t().contiguous()
        #data = data.view(data.data.shape[0], -1)
        #import pdb; pdb.set_trace()
        # data = Variable(train_in[i], volatile=True)
        # targets = Variable(train_tar[i])
        # print("Printing size in loop...")
        # print(data.shape)
        # print(targets.shape)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        # loss = criterion(output.view(-1, ntokens), targets)
        #targets = targets.view(targets.shape[0], 1)
        # import pdb; pdb.set_trace()
        loss = criterion(output.view(-1, ntokens), targets.long().view(-1))

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        _loss, correct = evaluate(train_in, train_tar.long())

        #if(i % 35 == 0 and batch > 0):
        # if batch % args.log_interval == 0 and batch > 0:
        cur_loss = total_loss[0] / args.log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} |  correct {:8.2f}'.format(
            epoch, batch, len(train_in) // args.bptt, lr,
            elapsed * 1000 / args.log_interval, cur_loss, correct))
        total_loss = 0
        start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        print("Starting training for epoch {}".format(epoch))
        train()
        #import pdb; pdb.set_trace()
        # val_loss = evaluate(val_data)
        val_loss, correct = evaluate(val_in, val_tar.long())
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'correct {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, correct))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, correct = evaluate(test_in, test_tar.long())
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
