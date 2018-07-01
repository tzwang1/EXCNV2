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
from collections import defaultdict
import os

import data
import conv_model as model

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM CNV detection Model')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data')
parser.add_argument('--data_in', type=str, default='data_x.pl',
                    help='location of the input training data')
parser.add_argument('--data_tar', type=str, default='data_y.pl',
                    help='location of the target training data')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--size', type=int, default=5,
                    help='size of data')
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
parser.add_argument('--show_every', type=int,  default=10,
                    help='how many epochs to train on before calculating training and val correct')
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

paths = {}
paths['data_in'] = os.path.join(args.data, args.data_in)
paths['data_tar'] = os.path.join(args.data, args.data_tar)

print("Getting input data from: {}".format(paths['data_in']))
print("Getting target data from: {}".format(paths['data_tar']))

#corpus = data.Corpus(paths)
corpus = data.Corpus(int(args.num), args.win_s, args.mini_win_s, paths, args.data)

def create_dict(input_data, target_data):
    d = defaultdict(list)
    for i in range(len(input_data)):
        d[len(input_data[i])].append((input_data[i], target_data[i]))
    
    return d

eval_batch_size = 10

data_in = corpus.data_x
data_tar = corpus.data_y

data_size = len(data_in)
num_train = int(0.8 * data_size)
# Split data into training and test
train_in, test_in = data_in[:num_train], data_in[num_train:]
train_tar, test_tar = data_tar[:num_train], data_tar[num_train:]

# Split training data into training and validation
num_train = int(0.8 * num_train)
train_in, val_in = train_in[:num_train], train_in[num_train:]
train_tar, val_tar = train_tar[:num_train], train_tar[num_train:]

print("=" * 89)
print("TARGET VALUE SPREAD")
unique, counts = np.unique(train_tar, return_counts=True)
print("TRAIN: {}".format(dict(zip(unique, counts))))

unique, counts = np.unique(val_tar, return_counts=True)
print("VALIDATION: {}".format(dict(zip(unique, counts))))

unique, counts = np.unique(test_tar, return_counts=True)
print("TEST: {}".format(dict(zip(unique, counts))))
print("=" * 89)

train_data = create_dict(train_in, train_tar)
val_data = create_dict(val_in, val_tar)
test_data = create_dict(test_in, test_tar)

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

input_height = len(train_in[0][0])
input_width = len(train_in[0][0][0])

hyperparameters = {}
hyperparameters['nhid'] = args.nhid
hyperparameters['fcout1'] = args.fcout1
hyperparameters['fcout2'] = args.fcout2
hyperparameters['decode1'] = args.decode1

model = model.RNNModel(args.model, ntokens, input_height, input_width, hyperparameters, args.nlayers, convNet_params, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = corpus.length
    correct = 0.0
    total = 0.0
    correct_gain = 0.0
    total_gain_cnv = 0.0
    correct_loss = 0.0
    total_loss_cnv = 0.0
    correct_neutral = 0.0
    total_neutral_cnv = 0.0

    for key in dataset:
        input_s, target_s = zip(*dataset[key])
        input_tensors = torch.stack([torch.FloatTensor(s) for s in input_s])
        #input_tensors = input_tensors.transpose(0,1).contiguous()
        target_tensors = torch.LongTensor(target_s)

        num_tensors = input_tensors.shape[0]
        #num_tensors = input_tensors.shape[1]
        
        num_batches = int(np.ceil(num_tensors / float(args.batch_size)))
        
        for i in range(num_batches):
            start = i * args.batch_size
            end = start + args.batch_size

            data = Variable(torch.stack(input_tensors[start:end]))
            targets = Variable(target_tensors[start:end])
            # The batch size may be different in each epoch
            BS = data.size(0)
            #BS = data.size(1)
            hidden = model.init_hidden(BS)
            output, hidden = model(data, hidden)
            
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets.long().view(-1)).data
            hidden = repackage_hidden(hidden)

            pred = output.data.max(1, keepdim=True)[1].squeeze(-1)
            for i in range(len(pred)):
                #import pdb; pdb.set_trace()
                cur_pred = int(pred[i])
                cur_tar = int(targets[i])
                if cur_pred == cur_tar:
                    if cur_pred == 0:
                        correct_gain+=1
                        total_gain_cnv+=1
                    elif cur_pred == 1:
                        correct_neutral+=1
                        total_neutral_cnv+=1
                    else:
                        correct_loss+=1
                        total_loss_cnv+=1
                else:
                    if cur_tar == 0:
                        total_gain_cnv+=1
                    elif cur_tar == 1:
                        total_neutral_cnv+=1
                    else:
                        total_loss_cnv+=1

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            total += output.shape[0]
     
    return total_loss[0] / len(dataset), correct/total, correct_gain/total_gain_cnv, correct_neutral/total_neutral_cnv, correct_loss/total_loss_cnv
    # return total_loss / len(dataset), correct/total

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = corpus.length
    # hidden = model.init_hidden(args.batch_size)
    
    for key in train_data:
        input_s, target_s = zip(*train_data[key])
        input_tensors = torch.stack([torch.FloatTensor(s) for s in input_s])
        target_tensors = torch.LongTensor(target_s)
        #input_tensors = input_tensors.transpose(0,1).contiguous()
        if key >= 29 :
            import pdb; pdb.set_trace()
        num_tensors = input_tensors.shape[0]
        #num_tensors = input_tensors.shape[1]
        num_batches = int(np.ceil(num_tensors / float(args.batch_size)))
        
        for i in range(num_batches):
            start = i * args.batch_size
            end = start + args.batch_size

            if(args.cuda):
                data = Variable(torch.stack(input_tensors[start:end]).cuda)
                targets = Variable(target_tensors[start:end].cuda)
            else:
                data = Variable(torch.stack(input_tensors[start:end]))
                targets = Variable(target_tensors[start:end])

            #targets = Variable(target_tensors)
            # The batch size may be different in each epoch
            BS = data.size(0)
            #BS = data.size(1)
            
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.init_hidden(BS)
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets.long().view(-1))
            #loss /= float(data.size(1))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)
            optimizer.step()

            total_loss += loss.data
            if i % args.log_interval == 0 and i > 0:
                _loss, correct, correct_gain, correct_neutral, correct_loss = evaluate(train_data)

                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} |{:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                        'loss {:5.8f} |  train correct {:8.5f} | sequence length {}'.format(
                    epoch, i, num_batches-1, lr,
                    elapsed * 1000 / args.log_interval, _loss, correct, key))
                total_loss = 0
                start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
val_loss = 0
val_loss_lst = []
val_correct_lst = []
train_loss_lst = []
# At any point you can hit Ctrl + C to break out of training early.
try:
    print(model)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        # print("Starting training for epoch {}".format(epoch))
        train()
        if(epoch % args.show_every == 0):
            train_loss, train_correct,train_correct_gain, train_correct_neutral, train_correct_loss = evaluate(train_data)
            val_loss, val_correct, val_correct_gain, val_correct_neutral, val_correct_loss = evaluate(val_data)
            val_loss_lst.append(val_loss)
            val_correct_lst.append(val_correct)
            print('-' * 89)
            print('| epoch {:3d} | lr {} | time: {:5.2f}s | train loss {:5.5f} |'
                    'train correct{:.2f}|'
                    'train gain correct{:8.2f}|'
                    'train neutral correct{:8.2f}|'
                    'train loss correct{:8.2f}|'.format(epoch,lr, (time.time() - epoch_start_time),
                                                train_loss, train_correct, train_correct_gain, train_correct_neutral, train_correct_loss))
            # print('| epoch {:3d} | lr {} | time: {:5.2f}s | train loss {:5.5f} |'
            #         ' train correct{:.2f} |'.format(epoch,lr, (time.time() - epoch_start_time),train_loss, train_correct))
            print('-' * 89)
            print('-' * 89)
            print('| epoch {:3d} | lr {} | time: {:5.2f}s | valid loss {:5.5f} |'
                    'val correct{:8.2f} | '
                    'val gain correct{:8.2f}| '
                    'val neutral correct{:8.2f}| '
                    'val loss correct{:8.2f}| '.format(epoch,lr, (time.time() - epoch_start_time),
                                                val_loss, val_correct, val_correct_gain, val_correct_loss, val_correct_loss))
            # print('| epoch {:3d} | lr {} | time: {:5.2f}s | valid loss {:5.5f} |'
            #         ' val correct{:8.2f} | '.format(epoch,lr, (time.time() - epoch_start_time),val_loss, val_correct))
                                                
            print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #lr /= 4
            #lr /=1.001
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
test_loss, correct, test_correct_gain, test_correct_neutral, test_correct_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test correct {:8.2f} | test gain correct {:8.2f} | test neutral correct {:8.2f} | test loss correct {:8.2f}'.format(
    test_loss, correct, test_correct_gain, test_correct_neutral, test_correct_loss))
print('=' * 89)
