import argparse
import data
from main import evaluate
from main import batchify

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM CNV prediction Model')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--train_in', type=str, default='./data/train_in.pl',
                    help='location of the input corpus')
parser.add_argument('--train_tar', type=str, default='./data/train_tar.pl',
                    help='location of the target corpus')
parser.add_argument('--val_in', type=str, default='./data/val_in.pl',
                    help='location of the input corpus')
parser.add_argument('--val_tar', type=str, default='./data/val_tar.pl',
                    help='location of the target corpus')
parser.add_argument('--test_in', type=str, default='./data/test_in.pl',
                    help='location of the input corpus')
parser.add_argument('--test_tar', type=str, default='./data/test_tar.pl',
                    help='location of the target corpus')
parser.add_argument('--num', type=int, default=200000000,
                    help='number of training examples')
parser.add_argument('--win_s', type=int, default=10000,
                    help='length of sequence')
parser.add_argument('--mini_win_s', type=int, default=1000,
                    help='length of small window')

paths = {}
paths['train_in'] = args.train_in
paths['train_tar'] = args.train_tar

paths['val_in'] = args.val_in
paths['val_tar'] = args.val_tar

paths['test_in'] = args.test_in
paths['test_tar'] = args.test_tar

corpus = data.Corpus(args.num, args.win_s, args.mini_win_s, paths)

train_in, train_gaps, train_tar = main.batchify(corpus.train_in, corpus.train_tar, args.batch_size)
val_in, val_gaps, val_tar = main.batchify(corpus.val_in, corpus.val_tar, args.batch_size)
test_in, test_gaps, test_tar = main.batchify(corpus.test_in, corpus.test_tar, args.batch_size)

with open(args.save, 'rb') as f:
    model = torch.load(f)
# Run on test data.
train_loss, correct = main.evaluate(train_in, train_gaps, train_tar.long())
print('=' * 89)
print('| End of training | train loss {:5.2f} | train correct {:8.2f}'.format(
    train_loss, correct))
print('=' * 89)

# Run on test data.
val_loss, correct = main.evaluate(val_in, val_gaps, val_tar.long())
print('=' * 89)
print('| End of training | val loss {:5.2f} | val correct {:8.2f}'.format(
    val_loss, correct))
print('=' * 89)

# Run on test data.
test_loss, correct = main.evaluate(test_in, test_gaps, test_tar.long())
print('=' * 89)
print('| End of training | test loss {:5.2f} | test correct {:8.2f}'.format(
    test_loss, correct))
print('=' * 89)