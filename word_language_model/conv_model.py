import torch
import torch.nn as nn
from torch.autograd import Variable

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

    def __init__(self, rnn_type, ntoken, ninp, nhid, fcinp, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        #import pdb; pdb.set_trace()
        #self.drop = nn.Dropout(dropout)
        # # TODO: add convlutional layer

        stride = 0
        padding = 0
        conv_out_channel = 5
        conv_kernel_size = 5
        conv_pooling_kernel = 2
        fc_inp_size = fcinp

        self.conv2d = nn.Conv2d(1, conv_out_channel, (conv_kernel_size, 2))
        self.maxpool = nn.MaxPool1d(conv_pooling_kernel)

        self.fc1 = nn.Linear(15, fc_inp_size)

        # self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = getattr(nn, rnn_type)(fc_inp_size+1, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.rnn = nn.RNN(fc_inp_size+1, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

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
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden, gap):
        input_list = []
        # x is in shape (num // batch x batch x seq_len x 2)  
        for i in range(len(x)):
            #import pdb; pdb.set_trace()
            input_ = x[i]
            input_ = input_.unsqueeze(1)
            input_ = self.conv2d(input_)
            input_ = input_.squeeze(-1)
            input_ = self.maxpool(input_)
            input_ = input_.view(input_.shape[0], -1)
            
            input_ = self.fc1(input_)
            
            # Concatenate gap
            gap_ = gap[i].unsqueeze(-1)
            input_ = torch.cat((input_, gap_), 1)
            input_list.append(input_.unsqueeze(0))
        
        input_ = torch.cat(input_list, 0)

        output, hidden = self.rnn(input_, hidden)
        #output = self.drop(output)
        # output = output[-1] # Take the last output
        # print(output.shape)
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.decoder(output)
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
