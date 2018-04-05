import pandas as pd
import argparse
import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--input', type=str, default='data/input.out',
                    help='location of the input file data')
parser.add_argument('--target', type=str, default='data/target.out',
                    help='location of the input file data')

args = parser.parse_args()

def clean(input_path):
    x = pd.read_csv(input_path, sep='\t', header=None)
    x[x.columns[0]] = x[x.columns[0]].map(lambda x: x.lstrip('chr'))
    x[x.columns[2]] = x[x.columns[2]].str.upper()
    # x[x.columns[3]] = x[x.columns[3]].astype(int)

    x.to_csv(input_path, sep='\t', index=False, header=False)


clean(args.input)
# clean(input_val, output_val)
# clean(input_test, output_test)

#clean(input_train, output_train)
#clean(input_val, output_val)
#clean(input_test, output_test)

