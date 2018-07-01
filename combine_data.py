import numpy as np
import pickle
import glob
import os
import argparse
import data

parser = argparse.ArgumentParser(description='Combine all input and target pickle files')
parser.add_argument('--data', type=str, default='data_to_transfer/data',
                    help='folder containing input and target pickle files that need to be combined')

args = parser.parse_args()

all_data_x = []
all_data_y = []

input_path = os.path.join(args.data, "*_input.pl")
target_path = os.path.join(args.data, "*_tar.pl")

input_data_list = glob.glob(input_path)
target_data_list = glob.glob(target_path)


for infile in sorted(input_data_list):
    print(infile)
    with open(infile) as f:
        data_x = pickle.load(f)
    all_data_x += data_x

for tarfile in sorted(target_data_list):
    print(tarfile)
    with open(tarfile) as f:
        data_y = pickle.load(f)
    all_data_y += data_y

data.save_data(all_data_x, all_data_y, os.path.join(args.data,'data_x.pl'), os.path.join(args.data,'data_y.pl'))


