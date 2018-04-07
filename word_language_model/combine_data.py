import numpy as np
import pickle
import glob
import os
import argparse
import data

parser = argparse.ArgumentParser(description='Combine all input and target pickle files')
parser.add_argument('--data', type=str, default='',
                    help='folder containing input and target pickle files that need to be combined')

args = parser.parse_args()

all_data_x = []
all_data_y = []

input_path = os.path.join(args.data, "*_input.pl")
target_path = os.path.join(args.data, "*_tar.pl")

input_data_list = glob.glob(input_path)
target_data_list = glob.glob(target_path)

for file in glob.glob(input_path):
    print(file)
    # data_x = pickle.load(file)
    # all_data_x += data_x

for file in glob.glob(target_path):
    print(file)
    # data_y = pickle.load(file)
    # all_data_y += data_y

#data.save_data(all_data_x, all_data_y, 'data_x.pl', 'data_y.pl')


