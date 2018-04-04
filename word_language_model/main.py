# coding: utf-8
import numpy as np
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
import conv_model as model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data_folder', type=str, default='data',
                    help='location of the data')
parser.add_argument('--train_in', type=str, default='train_in.pl',
                    help='location of the input training data')
parser.add_argument('--train_tar', type=str, default='train_tar.pl',
                    help='location of the target training data')
parser.add_argument('--val_in', type=str, default='val_in.pl',
                    help='location of the input validation data')
parser.add_argument('--val_tar', type=str, default='val_tar.pl',
                    help='location of the target validation data')
parser.add_argument('--test_in', type=str, default='test_in.pl',
                    help='location of the input testing data')
parser.add_argument('--test_tar', type=str, default='test_tar.pl',
                    help='location of the target testing data')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--size', type=int, default=5,
                    help='size of data')
# parser.add_argument('--emsize', type=int, default=200,
#                     help='size of word embeddings')
parser.add_argument('--num', type=int, default=200000000,
                    help='number of training examples')
parser.add_argument('--win_s', type=int, default=10000,
                    help='length of sequence')
parser.add_argument('--mini_win_s', type=int, default=100,
                    help='length of small window')
parser.add_argument('--nhid', type=int, default=50,
                    help='number of hidden units per layer')
parser.add_argument('--padding', type=int, default=0,
                    help='amount of padding for CNN')
parser.add_argument('--stride', type=int, default=1,
                    help='amount of stride for CNN')
parser.add_argument('--kernel_h', type=int, default=5,
                    help='kernel height for CNN')
parser.add_argument('--kernel_w', type=int, default=1,
                    help='kernel width for CNN')
parser.add_argument('--pool_kernel', type=int, default=2,
                    help='pool kernel size for CNN')
parser.add_argument('--out_channel', type=int, default=5,
                    help='number of output channels for CNN')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--fcout1', type=int, default=100,
                    help='fully connected layer 1 output size')
parser.add_argument('--fcout2', type=int, default=100,
                    help='fully connected layer 2 output size')
parser.add_argument('--decode1', type=int, default=100,
                    help='decoder output size')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='bptt length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
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
# num = 100000
# seq_len = 30
paths = {}
paths['train_in'] = args.data_folder + "/" + args.train_in
paths['train_tar'] = args.data_folder + "/" + args.train_tar

paths['val_in'] = args.data_folder + "/" + args.val_in
paths['val_tar'] = args.data_folder + "/" + args.val_tar

paths['test_in'] = args.data_folder + "/" + args.test_in
paths['test_tar'] = args.data_folder + "/" + args.test_tar

corpus = data.Corpus(int(args.num), args.win_s, args.mini_win_s, paths, args.data_folder)

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

def batchify(input_data_, target_data, bsz):
    print("Batchifying data....")
    input_gaps = [data[0] for data in input_data_]
    input_data = [data[1] for data in input_data_]

    input_gaps = torch.FloatTensor(input_gaps)
    input_data = torch.FloatTensor(input_data)
    target_data = torch.FloatTensor(target_data)

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = input_data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    input_data = input_data.narrow(0, 0, nbatch * bsz)
    input_gaps = input_gaps.narrow(0, 0, nbatch * bsz)
    target_data = target_data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    input_data = input_data.view(input_data.shape[0] // bsz, bsz, input_data.shape[1], -1)
    input_gaps = input_gaps.view(input_gaps.shape[0] // bsz, bsz)
    target_data = target_data.view(bsz, -1).t().contiguous()
    # if args.cuda:
    #     input_data = input_data.cuda()
    #     target_data = target_data.cuda()
    return input_data, input_gaps, target_data


eval_batch_size = 10

# train_dataset = corpus.train_dataset
# val_dataset = corpus.val_dataset
# test_dataset = corpus.test_dataset

# unique, counts = np.unique(corpus.train_tar, return_counts=True)
# weights = 1/torch.from_numpy(counts).float()
# weights = weights.double()
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, args.batch_size)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, sampler = sampler)

train_in, train_gaps, train_tar = batchify(corpus.train_in, corpus.train_tar, args.batch_size)
val_in, val_gaps, val_tar = batchify(corpus.val_in, corpus.val_tar, args.batch_size)
test_in, test_gaps, test_tar = batchify(corpus.test_in, corpus.test_tar, args.batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = corpus.length

convNet_params = {}
convNet_params['padding'] = args.padding
convNet_params['stride'] = args.stride
convNet_params['kernel_h'] = args.kernel_h
convNet_params['kernel_w'] = args.kernel_w
convNet_params['out_channel'] = args.out_channel
convNet_params['pool_kernel'] = args.pool_kernel
# conv = model.ConvNet()
input_height = train_in.shape[2]
input_width = train_in.shape[3]

hyperparameters = {}
hyperparameters['nhid'] = args.nhid
hyperparameters['fcout1'] = args.fcout1
hyperparameters['fcout2'] = args.fcout2
hyperparameters['decode1'] = args.decode1

model = model.RNNModel(args.model, ntokens, input_height, input_width, hyperparameters, args.nlayers, convNet_params, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
# optimizer_conv = torch.optim.Adam(conv.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

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

def get_batch(input_data, gap_data, target_data, i, evaluation=False):
    sequence_len = min(args.bptt, len(input_data) - 1 - i)
    
    data = input_data[i:i+sequence_len]
    gap = gap_data[i:i+sequence_len]
    target = target_data[i:i+sequence_len]
    # if args.cuda:
    #     data = data.cuda()
    #     target = target.cuda()

    data = Variable(data, volatile=evaluation)
    gap = Variable(gap, volatile=evaluation)
    target = Variable(target)

    return data, gap, target

def evaluate(input_data, gap_data, target_data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = corpus.length
    correct = 0.0
    total = 0.0
    hidden = model.init_hidden(args.batch_size)
    # hidden = model.init_hidden(eval_batch_size)
    # for i in range(0, data_source.size(0) - 1, args.bptt)
    for batch, i in enumerate(range(0, input_data.size(0) - 1, args.bptt)):
        data, gap, targets = get_batch(input_data, gap_data, target_data, i)

        output, hidden = model(data, hidden, gap.float())
        
        total_loss += len(data) * criterion(output.view(-1, ntokens), targets.long().view(-1)).data
        hidden = repackage_hidden(hidden)

        pred = output.data.max(2, keepdim=True)[1].squeeze(-1)
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
        total += output.shape[0] * output.shape[1]
    return total_loss[0] / len(input_data), correct/total

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = corpus.length
    hidden = model.init_hidden(args.batch_size)
    
    for batch, i in enumerate(range(0, train_in.size(0) - 1, args.bptt)):
        # print("Starting training iteration {}".format(i))
        data, gap, targets = get_batch(train_in, train_gaps, train_tar, i)
    # data_list = []
    # gap_list = []
    # target_list = []
    # for sample in train_loader:
    #     data_list.append(sample['input'].float())
    #     gap_list.append(sample['gap'].float())
    #     target_list.append(sample['target'].float())
    #     import pdb; pdb.set_trace()
    #     if(len(data_list) != args.bptt):
    #         continue
    #     else:
    #         data = Variable(torch.cat(data_list))
    #         gap = Variable(torch.cat(gap_list))
    #         targets = Variable(torch.cat(target_list))
    #         data_list, gap_list, target_list = []
        
    # for sample in train_loader:
    #     data = Variable(sample['input'].float())
    #     gap = Variable(sample['gap'].float())
    #     targets = Variable(sample['target'])
        #import pdb; pdb.set_trace()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, gap)
        loss = criterion(output.view(-1, ntokens), targets.long().view(-1))

        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        if batch % args.log_interval == 0 and batch > 0:
            _loss, correct = evaluate(train_in, train_gaps, train_tar.long())

            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                    'loss {:5.8f} |  train correct {:8.5f}'.format(
                epoch, batch, len(train_in) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, correct))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
val_loss_lst = []
val_correct = []
train_loss_lst = []
# At any point you can hit Ctrl + C to break out of training early.
try:
    print(model)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        # print("Starting training for epoch {}".format(epoch))
        train()
        #import pdb; pdb.set_trace()
        # val_loss = evaluate(val_data)
        val_loss, correct = evaluate(test_in, test_gaps, test_tar.long())
        val_loss_lst.append(val_loss)
        val_correct.append(correct)
        print('-' * 89)
        print('| end of epoch {:3d} | lr {} |  time: {:5.2f}s | valid loss {:5.5f} | '
                'val correct {:8.2f}'.format(epoch,lr, (time.time() - epoch_start_time),
                                             val_loss, correct))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            # lr /= 1.2
            lr = lr
    fig, ax = plt.subplots( nrows=1, ncols=1)
    ax.plot(val_correct)
    fig.savefig('val_correct.png')
    plt.close(fig)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, correct = evaluate(test_in, test_gaps, test_tar.long())
print('=' * 89)
print('| End of training | test loss {:5.2f} | test correct {:8.2f}'.format(
    test_loss, correct))
print('=' * 89)
