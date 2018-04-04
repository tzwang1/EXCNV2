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

    def __init__(self, rnn_type, ntoken, input_height, input_width, hyperparameters, nlayers, cnn_params, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        #import pdb; pdb.set_trace()
        stride = cnn_params['stride']
        padding = cnn_params['padding']
        conv_out_channel = cnn_params['out_channel']
        conv_kernel_height = cnn_params['kernel_h']
        conv_kernel_width = cnn_params['kernel_w']
        conv_pooling_kernel = cnn_params['pool_kernel']
        
        ninp = input_height * input_width
        
        nhid = hyperparameters['nhid']
        fc1_out_size = hyperparameters['fcout1']
        fc2_out_size = hyperparameters['fcout2']
        decode1_out = hyperparameters['decode1']

        conv1_out_height= int(((input_height - conv_kernel_height + 2*padding)/stride + 1)/conv_pooling_kernel)
        conv1_out_width = int(((input_width - conv_kernel_width + 2*padding)/stride + 1)/conv_pooling_kernel)

        conv2_out_height = int(((conv1_out_height - conv_kernel_height + 2*padding)/stride + 1)/conv_pooling_kernel)
        # conv2_out_width = int(((conv1_out_width - conv_kernel_width + 2*padding)/stride + 1)/conv_pooling_kernel)
        conv2_out_width = conv1_out_width

        conv1_out_size = int(conv_out_channel * conv1_out_height * conv1_out_width)
        conv2_out_size = int(conv_out_channel*2 * conv2_out_height * conv2_out_width)

        self.conv2d1 = nn.Conv2d(1, conv_out_channel, (conv_kernel_height, conv_kernel_width))
        self.maxpool1 = nn.MaxPool2d(conv_pooling_kernel)
        self.conv2d2 = nn.Conv2d(conv_out_channel, conv_out_channel*2, (conv_kernel_height, 1))
        self.maxpool2 = nn.MaxPool1d(conv_pooling_kernel)

        #import pdb; pdb.set_trace()
        # self.fc1 = nn.Linear(conv1_out_size, fc1_out_size)
        self.fc1 = nn.Linear(conv2_out_size, fc1_out_size)
        self.fc2 = nn.Linear(fc1_out_size, fc2_out_size)

        if rnn_type in ['LSTM', 'GRU']:
            # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = getattr(nn, rnn_type)(fc1_out_size+1, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.rnn = nn.RNN(fc_inp_size+1, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        # self.decoder = nn.Linear(nhid, ntoken)
        self.decoder1 = nn.Linear(nhid, decode1_out)
        self.decoder2 = nn.Linear(decode1_out, ntoken)

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
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        input_list = []
        # x is in shape (num // batch x batch x seq_len x 2) 
        # BPTT IMPLEMENTATION
        #import pdb; pdb.set_trace()
        for i in range(len(x)):
            input_ = x[i]
            input_ = input_.unsqueeze(1)
            input_ = self.conv2d1(input_)
            input_ = input_.squeeze(-1)
            input_ = self.maxpool1(input_)
            # import pdb; pdb.set_trace()
            # input_ = input_.squeeze(-1)
            input_ = self.conv2d2(input_)
            input_ = input_.squeeze(-1)
            input_ = self.maxpool2(input_)
            input_ = input_.view(input_.shape[0], -1)
            
            input_ = self.fc1(input_)
            input_ = self.fc2(input_)
            
            # Concatenate gap
            # gap_ = gap[i].unsqueeze(-1)
            # input_ = torch.cat((input_, gap_), 1)
            input_list.append(input_.unsqueeze(0))
    
        input_ = torch.cat(input_list, 0)
        
        #import pdb; pdb.set_trace()
        output, hidden = self.rnn(input_, hidden)
        # output = self.drop(output)
        # output = output[-1] # Take the last output
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
