import pandas as pd
import argparse
import data
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data_list', type=str, default='data/input.out',
                    help='location of the input file data')
parser.add_argument('--clean', type=str, default='True',
                    help='clean data')

args = parser.parse_args()
data_list = pd.read_csv(args.data_list, sep=",", header=None)

def clean(input_path):
    print("Cleaning {}".format(input_path))
    x = pd.DataFrame()
    with open(input_path) as f:
        for line in f:
            line = line.strip().split('\t')
            if(len(line) == 4):
                line[0] = line[0].lstrip('chr')
                x = pd.concat([x, pd.DataFrame([line])], ignore_index=True)

    x.to_csv(input_path, sep='\t', index=False, header=False)

num_cores = multiprocessing.cpu_count()

print("Cleaning data")
if(args.clean == 'True'):
    Parallel(n_jobs=num_cores)(delayed(clean)(data_list[0][i]) for i in range(len(data_list)))

all_data_x = []
all_data_y = []
num = -1
window_size = 10000
mini_window_size = 100

print("Processing data into inputs and targets")
all_data = Parallel(n_jobs=num_cores)(delayed(data.load_data)(data_list[0][i], data_list[1][i], num, window_size, mini_window_size) for i in range(len(data_list)))

for data_item in all_data:
    all_data_x += data_item[0]
    all_data_y += data_item[1]

data.save_data(all_data_x, all_data_y, 'data_x.pl', 'data_y.pl')

