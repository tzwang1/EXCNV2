import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        # # TODO: add convlutional layer
        # self.downconv1 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=5, padding=5 // 2),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU())

        # self.upconv1 = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2),
        #     nn.Upsample(scale_factor=2),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU())
        # self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        #self.sigmoid = nn.Sigmoid()

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

    def forward(self, input, hidden):
        # import pdb; pdb.set_trace()
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
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
