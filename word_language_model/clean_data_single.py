import pandas as pd
import argparse
import data
from joblib import Parallel, delayed
import multiprocessing
import os

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--input_data', type=str, default='',
                    help='input file data')
parser.add_argument('--target_data', type=str, default='',
                    help='target file data')
parser.add_argument('--clean', type=str, default='True',
                    help='clean data')
parser.add_argument('--output_dir', type=str, default='clean_pickle',
                    help='output folder for pickle files')

args = parser.parse_args()
#data_list = pd.read_csv(args.data_list, sep=",", header=None)

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


all_data_x = []
all_data_y = []
num = -1
window_size = 10000
mini_window_size = 100

print("Processing data into inputs and targets")
#all_data = Parallel(n_jobs=num_cores)(delayed(data.load_data)(data_list[0][i], data_list[1][i], num, window_size, mini_window_size) for i in range(len(data_list)))

all_data_x, all_data_y = data.load_data(args.input_data, args.target_data, num, window_size, mini_window_size)


input_path = os.path.join(args.output_dir, args.input_data.replace("out", "pl"))
target_path = os.path.join(args.output_dir, args.target_data.replace("out", "pl"))
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
data.save_data(all_data_x, all_data_y, input_path, target_path)

