import torch
import torch.nn as nn
from torch.autograd import Variable
import math

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         conv_out_channel = 5
#         conv_kernel_size = 5
#         conv_pooling_kernel = 2
#         fc_inp_size = 10

#         self.conv2d = nn.Conv2d(1, conv_out_channel, (conv_kernel_size, 2))
#         self.maxpool = nn.MaxPool1d(conv_pooling_kernel)

#         self.fc1 = nn.Linear(15, fc_inp_size)

#     def forward(self, input_):
#         # output = self.conv1d(input)
#         # Apply 2D Conv and 1D maxpool
#         input_ = input_.unsqueeze(1)
#         input_ = self.conv2d(input_)
#         input_ = input_.squeeze(-1)
#         input_ = self.maxpool(input_)
#         input_ = input_.view(input_.shape[0], -1)
        
#         output = self.fc1(input_)

#         return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, input_height, input_width, hyperparameters, nlayers, cnn_params, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.stride = cnn_params['stride']
        self.padding = cnn_params['padding']
        self.conv_out_channel = cnn_params['out_channel']
        self.conv_kernel_height = cnn_params['kernel_h']
        self.conv_kernel_width = cnn_params['kernel_w']
        self.conv_pooling_kernel = cnn_params['pool_kernel']
        self.input_height = input_height
        self.input_width = input_width
        self.ninp = self.input_height * self.input_width
        
        self.nhid = hyperparameters['nhid']
        self.fc1_out_size = hyperparameters['fcout1']
        self.fc2_out_size = hyperparameters['fcout2']
        self.decode1_out = hyperparameters['decode1']

        self.conv1_out_height= int(((self.input_height - self.conv_kernel_height + 2*self.padding)/self.stride + 1)/self.conv_pooling_kernel)
        self.conv1_out_width = int(((self.input_width - self.conv_kernel_width + 2*self.padding)/self.stride + 1)/self.conv_pooling_kernel)

        self.conv2_out_height = int(((self.conv1_out_height - self.conv_kernel_height + 2*self.padding)/self.stride + 1)/self.conv_pooling_kernel)
        # conv2_out_width = int(((conv1_out_width - conv_kernel_width + 2*padding)/stride + 1)/conv_pooling_kernel)
        self.conv2_out_width = self.conv1_out_width

        self.conv1_out_size = int(self.conv_out_channel * self.conv1_out_height * self.conv1_out_width)
        self.conv2_out_size = int(self.conv_out_channel*2 * self.conv2_out_height * self.conv2_out_width)

        self.conv2d1 = nn.Conv2d(1, self.conv_out_channel, (self.conv_kernel_height, self.conv_kernel_width))
        self.maxpool1 = nn.MaxPool2d(self.conv_pooling_kernel)
        self.conv2d2 = nn.Conv2d(self.conv_out_channel, self.conv_out_channel*2, (self.conv_kernel_height, 1))
        self.maxpool2 = nn.MaxPool1d(self.conv_pooling_kernel)

        #import pdb; pdb.set_trace()
        # self.fc1 = nn.Linear(conv1_out_size, fc1_out_size)
        self.fc1 = nn.Linear(self.conv2_out_size, self.fc1_out_size)
        self.fc2 = nn.Linear(self.fc1_out_size, self.fc2_out_size)

        if rnn_type in ['LSTM', 'GRU']:
            # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = getattr(nn, rnn_type)(self.fc2_out_size, self.nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.rnn = nn.RNN(fc2_out_size, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        # self.decoder = nn.Linear(nhid, ntoken)
        self.decoder1 = nn.Linear(self.nhid, self.decode1_out)
        self.decoder2 = nn.Linear(self.decode1_out, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        input_list = []
        # x is in shape batch_size x seq_len x mini_window_size x 2
        #Switching order of input_tensors to be seq_len x batch_size x mini_window x batch_size
        x = x.transpose(0,1).contiguous()
        # x is in shape seq_len x batch_size x mini_window_size x 2
        for i in range(len(x)):
            input_ = x[i]
            input_ = input_.unsqueeze(1)
            input_ = self.conv2d1(input_)
            input_ = input_.squeeze(-1)
            input_ = self.maxpool1(input_)
            # input_ = input_.squeeze(-1)
            input_ = self.conv2d2(input_)
            input_ = input_.squeeze(-1)
            input_ = self.maxpool2(input_)
            input_ = input_.view(input_.shape[0], -1)
            
            input_ = self.fc1(input_)
            input_ = self.fc2(input_)
            
            input_list.append(input_.unsqueeze(0))
    
        input_ = torch.cat(input_list, 0)
        output, hidden = self.rnn(input_, hidden)
        output = self.drop(output)
        output = output[-1] # Take the last output
        # print(output.shape)
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.decoder1(output)
        decoded = self.decoder2(decoded)
        # print(decoded.shape)
        #decoded = self.sigmoid(decoded)
        return decoded, hidden
        #return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        # return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            #print(weight.new(self.nlayers, bsz, self.nhid).zero_().long().type())
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
