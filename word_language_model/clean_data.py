import pandas as pd
import argparse
import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data_list', type=str, default='data/input.out',
                    help='location of the input file data')

args = parser.parse_args()
data = pd.read_csv(args.data_list, sep=",", header=None)

def clean(input_path):
    x = pd.read_csv(input_path, sep='\t', header=None)
    x[x.columns[0]] = x[x.columns[0]].map(lambda x: x.lstrip('chr'))
    x[x.columns[2]] = x[x.columns[2]].str.upper()
    # x[x.columns[3]] = x[x.columns[3]].astype(int)

    x.to_csv(input_path, sep='\t', index=False, header=False)

for i in range(len(data)):
    clean(data[i][0])

all_data_x = []
all_data_y = []

for i in range(len(data)):
    data_x, data_y = data.load_data_from_file(data[i][0], data[i][1])
    all_data_x += data_x
    all_data_y += data_y

data.save(all_data_x, all_data_y, 'data_x.pl', 'data_y.pl')

# clean(input_val, output_val)
# clean(input_test, output_test)

#clean(input_train, output_train)
#clean(input_val, output_val)
#clean(input_test, output_test)

