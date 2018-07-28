import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math

all_bases = ['A', 'T', 'C', 'G']
n_bases = len(all_bases)

all_targets = ['gain', 'neutral', 'loss']
n_targets = len(all_targets)

new_all_targets = ['Yes', 'No']

# Finds base index from all_bases
def base_to_index(base):
    return all_bases.index(base)

# Finds target index from all_targets
def target_to_index(target):
    return all_targets.index(target)

def new_target_to_index(target):
    return new_all_targets.index(target)

# Converts base to one hot encoding representation
def base_to_one_hot(base):
    '''
    Args:
        base = nucleotide
    Returns:
        numpy array of a one hot encoding representation
    '''
    base.upper()
    #tensor = np.zeros((1, n_bases))
    tensor = torch.zeros((1, n_bases))
    tensor[0][base_to_index(base)] = 1
    return tensor

def seq_to_tensor(seq):
    seq.upper()
    # tensor = np.zeros((len(seq), 1, n_bases))
    tensor = torch.zeros((len(seq), 1, n_bases))
    for i, base in enumerate(seq):
        tensor[i][0][base_to_index(base)] = 1
    
    return tensor

# Converts target to one hot encoding representation
def target_to_one_hot(target):
    '''
    Args:
        target = type of CNV
    Returns:
        torch tensor of a one hot encoding representation
    '''
    # tensor = np.array([all_targets.index(target)])
    tensor = torch.FloatTensor([all_targets.index(target)])

    return tensor


def calculate_mini_window_feature(mini_window):
    #import pdb; pdb.set_trace()
    mini_window = np.array(mini_window)
    try:
        depth = np.mean(mini_window[:, -1].astype(float))
    except:
        import pdb; pdb.set_trace()
    gc_num = len(np.where((mini_window[:, -2] == 'C') | (mini_window[:, -2] == 'G')| (mini_window[:, -2] == 'c') | (mini_window[:, -2] == 'g'))[0])
    gc_ratio = gc_num * 1.0 / len(mini_window)
    if math.isnan(depth) or math.isnan(gc_ratio):
        import pdb; pdb.set_trace()

    return [depth, gc_ratio]

def calculate_target_features(chrom, start, targets):
    #import pdb; pdb.set_trace()
    found = False
    for i in range(len(targets)):
        try:
            tar_chrom = int(targets[i][0])
        except:
            continue
        if(chrom == tar_chrom):
            if(start >= targets[i][1] and start < targets[i][2]):
                target = target_to_index(targets[i][3])
                found = True
                break
    if not found:
        target = 3

    return target


def calculate_mini_window(window, mini_window_size):
    #import pdb; pdb.set_trace()
    features = []
    for i in range(len(window) / mini_window_size):
        start, end = i * mini_window_size, (i + 1) * mini_window_size
        feature = calculate_mini_window_feature(window[start: end])
        features.append(feature)
    # return [int(window[0][1]), np.array(features)]
    return np.array(features)

def process_targets(targets):
    new_targets = []
    for i in range(len(targets)-1):
        if targets[i][0] == targets[i+1][0]:
            new_start = targets[i][1]
            new_end = targets[i+1][2]
            if targets[i][3] == targets[i+1][3]:
                breakpoint = "No"
            else:
                breakpoint = "Yes"

            new_targets.append([targets[i][0], new_start, new_end, breakpoint])

    return new_targets

def get_valid_chrom(targets):
    valid = set()
    for target in targets:
        if target[0] not in valid:
            valid.add(target[0])

    return valid

def load_new(num, input_path, target_path, window_size, mini_window_size):
    '''
    New style of data preprocessing
    Identifies breakpoints in between segments.
    Two segments with the same target do not contain a break point, 
    while two segments with different breakpoints contain a break point.

    '''
    window_len = window_size / mini_window_size
    if num == -1:
        chunksize=-1
    else:
        chunksize=num

    targets = np.genfromtxt(target_path, dtype=None, encoding=None)
    targets = process_targets(targets)
    
    valid_chrom = get_valid_chrom(targets)

    print("Reading in chunks of size {}".format(chunksize))
    print("Processing input into windows of size {}".format(window_size))
    print("Processing input in mini_windows of size {}".format(mini_window_size))

    cur_target = 0
    all_windows = []
    all_targets = []
    window = []
    mini_window = []

    for chunk in pd.read_csv(input_path, delimiter="\t", chunksize=chunksize, names='abcd', dtype=str):
        cur_target = 0
        input_ = np.asarray(chunk)
        for row in input_:
            if row[0] not in valid_chrom:
                continue
           
            # Try and match the current rows chrom with the targets chrom
            while row[0] != targets[cur_target][0]:
                if row[0] > targets[cur_target][0]:
                    if cur_target < len(targets)-1:
                        cur_target+=1
                break
            
            # if chrom of current row is less than the chrom of the target, go to the next row
            if row[0] < targets[cur_target][0]:
                continue

            # Check whether the current position is within the start and end of the target
            if row[1] >= targets[cur_target][0] and row[1] <= targets[cur_target][1]:
                mini_window.append([row[2], row[3]])
                if len(mini_window) == mini_window_size:
                    mini_window_features = calculate_mini_window_feature(mini_window)
                    window.append(mini_window_features)
                    mini_window = []
                    if len(window) == window_len:
                        print("Appending to all windows")
                        all_windows.append(window)
                        all_targets.append(new_target_to_index(targets[cur_target][3]))
                        window = []
            elif row[1] > targets[cur_target][1]:
                if cur_target < len(targets)-1:
                    cur_target+=1
                    # throws away old target if new targets breakpoint type is different
                    if targets[cur_target][3] != targets[cur_target-1][3]:
                        window = []
                        mini_window = []
                            
    return all_windows, all_targets



def load(num, input_path, target_path, window_size, mini_window_size):
    '''
    Old style of data preprocessing 
    Separates input data in segments of size window_size,
    with mini windows of size mini_window_size. Each mini_window contains the average read depth
    and average GC content of size mini_window_size bases.
    '''
    MAX_LARGE_WINDOW_SIZE = 10**6

    if(num == -1):
        #input_ = pd.read_csv(input_path, delimiter="\t",names="abcd")
        chunksize=None
    else:
        #input_ = pd.read_csv(input_path, delimiter="\t", nrows=num, names="abcd")
        chunksize=num
    
    targets = np.genfromtxt(target_path, dtype=None, encoding=None)
    targets2 = pd.read_csv(target_path, dtype=str, delimiter="\t")
    targets2 = np.asarray(targets2)
    #input_ = np.asarray(input_)
    
    all_s_features = []
    all_window_targets = []
    
    print("Reading in chunks of size {}".format(chunksize))
    for chunk in pd.read_csv(input_path, delimiter="\t", chunksize=chunksize, names="abcd", dtype=str):
        input_ = np.asarray(chunk)
        small_targets = []
        for i in range(len(targets)):
            try:
                if(int(targets[i][2]) - int(targets[i][1]) < MAX_LARGE_WINDOW_SIZE):
                    small_targets.append(targets[i])
            except:
                continue
        targets = np.array(small_targets)
        tar_pos = 0
        max_tar_pos = len(targets)
        s = []
        windows = []
        tmp_window = []
        count = 0
        windows_targets = []
        for i in range(len(input_)):
            chrom = targets[tar_pos][0]
            end_point = int(targets[tar_pos][2])
            try:
                if int(input_[i,1]) <= end_point:
                    tmp_window.append(input_[i])
                    count += 1
                    if count == window_size:
                        windows.append(tmp_window)
                        tmp_window, count = [], 0
                        if len(windows) > 100:
                            windows = []
                            continue
                else:
                    tmp_window.append(input_[i])
                    count += 1
                    if count == window_size:
                        windows.append(tmp_window)
                    if len(windows) > 0 and len(windows) <= 100:
                        s.append(windows)
                        windows_targets.append(target_to_index(targets[tar_pos][3]))
                    windows, tmp_window, count = [], [], 0
                    if(tar_pos < len(targets)-1):
                        tar_pos+=1
                    else:
                        break
            except:
                print("Error processing value {}".format(input_[i,1]))
                continue
        s_features = []
        # windows_targets = []
        for k in range(len(s)):
            windows = s[k]
            windows_features = []
            for i in range(len(windows)):
                #print("Looking at window {} out of {} total".format(i, len(windows)))
                chrom = windows[i][0][0]
                features = calculate_mini_window(windows[i], mini_window_size)
                windows_features.append(features)
            s_features.append(windows_features)
        
        all_s_features += s_features
        all_window_targets += windows_targets
    return all_s_features, all_window_targets

def load_data(input_path, target_path, num, window_size, mini_window_size):
    '''
    Args:
        input_path: file path to input file containing read depth, base
        target_path: file path to target file containing CNV info (Yes, No)
        num: number of training/test/validation examples
        seq_len: length of sequence to input

    Returns:
        x: input matrix of size (num x batch_size x seq_len x 5)
        y: target matrix of size (num x 1)

    '''
   
    print("Processing data for {} and {}...".format(input_path, target_path))
    x, y = load_new(num, input_path, target_path, window_size, mini_window_size)
         
    return x, y

def save_data(input_data, target_data, in_out_path, tar_out_path):
    '''
    Args:
        input_data: input array of size (num training x batch size (optional) x seq len x 5)
        target_data: target array of size (num training x batch size (optional) x 1)
        in_out_path: path to save the input file to
        tar_out_path: path to save the output file to
    Returns:

    '''
    with open(in_out_path, 'wb+') as in_out:
        pickle.dump(input_data, in_out)

    with open(tar_out_path, 'wb+') as tar_out:
        pickle.dump(target_data, tar_out)

def load_data_from_file(in_path, tar_path):
    '''
    Args:
        in_path: path to input file
        tar_path: path to target file
    
    Returns:
        x: numpy array of input data
        y: numpy array of target data
    '''
    print("Loading data from file...")
    # x = np.load(in_path)
    # y = np.load(tar_path)

    with open(in_path, 'r') as infile:
        x = pickle.load(infile)
    
    with open(tar_path, 'r') as tarfile:
        y = pickle.load(tarfile)

    return x, y

def rearrange(input_data):
    '''
    Args: 
        input_data: input data in the form of batch size x seq_len x input size
    
    Returns:
        new_input_data: input data in the form of seq_len x batch_size x input size
    '''

    new_input_data = torch.zeros((input_data.shape[1], input_data.shape[0], input_data.shape[2]))

    for i in range(0, input_data.shape[1]): # iterate through seq
        for j in range(0, input_data.shape[0]): # iterate through batches
            new_input_data[i][j] = input_data[j][i]
    
    return new_input_data

class CNVdata(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = [data[1] for data in input_data]
        self.input_gaps = [data[0] for data in input_data]
        self.target_data = target_data
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        sample ={'input': self.input_data[idx], 'gap': self.input_gaps[idx], 'target': self.target_data[idx]}
        return sample


class Corpus(object):
    # def __init__(self, path):
    def __init__(self, num, window_size, mini_window_size, paths, data_folder):
        self.length = n_targets

        # train_in_txt = data_folder + "/input_train.out"
        # train_tar_txt = data_folder + "/target_train.out"

        # val_in_txt = data_folder + "/input_val.out"
        # val_tar_txt = data_folder + "/target_val.out"

        # test_in_txt = data_folder + "/input_test.out"
        # test_tar_txt = data_folder + "/target_test.out"

        # train_in_path = paths['train_in']
        # train_tar_path = paths['train_tar']

        # val_in_path = paths['val_in']
        # val_tar_path = paths['val_tar']

        # test_in_path = paths['test_in']
        # test_tar_path = paths['test_tar']

        data_in_path = paths['data_in']
        data_tar_path = paths['data_tar']
        try:
            data_x, data_y = load_data_from_file(data_in_path, data_tar_path)
        except:
            print("Could not load presaved training data")
            # train_,x train_y = load_data(train_in_txt, train_tar_txt, num, window_size, mini_window_size)
            # save_data(train_x, train_y, train_in_path, train_tar_path)

        # try:
        #     train_x, train_y = load_data_from_file(train_in_path, train_tar_path)
        # except:
        #     print("Could not load presaved training data")
        #     train_x, train_y = load_data(train_in_txt, train_tar_txt, num, window_size, mini_window_size)
        #     save_data(train_x, train_y, train_in_path, train_tar_path)
        
        # try:
        #     val_x, val_y = load_data_from_file(val_in_path, val_tar_path)
        # except:
        #     print("Could not load presaved validation data")
        #     val_x, val_y = load_data(val_in_txt, val_tar_txt, num, window_size, mini_window_size)
        #     save_data(val_x, val_y, val_in_path, val_tar_path)

        # try:
        #     test_x, test_y = load_data_from_file(test_in_path, test_tar_path)
        # except:
        #     print("Could not load presaved test data")
        #     test_x, test_y = load_data(test_in_txt, test_tar_txt, num, window_size, mini_window_size)
        #     save_data(test_x,test_y, test_in_path, test_tar_path)
        print("TARGET VALUES")
        unique, counts = np.unique(data_y, return_counts=True)
        print("TRAIN TAR: {}".format(dict(zip(unique, counts))))

        # unique, counts = np.unique(val_y, return_counts=True)
        # print("VAL TAR: {}".format(dict(zip(unique, counts))))

        # unique, counts = np.unique(test_y, return_counts=True)
        # print("TEST TAR: {}".format(dict(zip(unique, counts))))

        # self.train_dataset = CNVdata(train_x, train_y)
        # self.val_dataset = CNVdata(val_x, val_y)
        # self.test_dataset = CNVdata(test_x, test_y)
        # validation_set = create_dict(val_x, val_y)
        # test_set = create_dict(test_x, test_y)

        self.data_x = data_x
        self.data_y = data_y

        # self.train_in = train_x
        # self.train_tar = train_y

        # self.val_in = train_x
        # self.val_tar = train_y

        # self.test_in = train_x
        # self.test_tar = train_y
    
        # self.val_in = val_x
        # self.val_tar = val_y

        # self.test_in = test_x
        # self.test_tar = test_y


        

    
