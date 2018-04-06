import pandas as pd
import argparse
import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data_list', type=str, default='data/input.out',
                    help='location of the input file data')
parser.add_argument('--clean', type=str, default='True',
                    help='clean data')

args = parser.parse_args()
data_list = pd.read_csv(args.data_list, sep=",", header=None)

def clean(input_path):
    x = pd.read_csv(input_path, sep='\t', header=None)
    x[x.columns[0]] = x[x.columns[0]].map(lambda x: x.lstrip('chr'))
    x[x.columns[2]] = x[x.columns[2]].str.upper()
    # x[x.columns[3]] = x[x.columns[3]].astype(int)

    x.to_csv(input_path, sep='\t', index=False, header=False)

if(args.clean == 'True'):
    print("Cleaning data")
    try:
        for i in range(len(data_list)):
            clean(data_list[0][i])
    except:
        print("data already cleaned")

all_data_x = []
all_data_y = []
num = -1
window_size = 10000
mini_window_size = 100
for i in range(len(data_list)):
    data_x, data_y = data.load_data(data_list[0][i], data_list[1][i], num, window_size, mini_window_size)
    all_data_x += data_x
    all_data_y += data_y

data.save_data(all_data_x, all_data_y, 'data_x.pl', 'data_y.pl')

# clean(input_val, output_val)
# clean(input_test, output_test)

#clean(input_train, output_train)
#clean(input_val, output_val)
#clean(input_test, output_test)

