import os
import torch
import numpy as np

all_bases = ['A', 'T', 'C', 'G']
n_bases = len(all_bases)

all_targets = ['No', 'Yes']
n_targets = len(all_targets)

# Finds base index from all_bases
def base_to_index(base):
    return all_bases.index(base)

# Finds target index from all_targets
def target_to_index(target):
    return all_targets.index(target)

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

def load(num, input_path, target_path, seq_len):
    x = np.zeros((num, seq_len, 5))
    y = np.zeros((num, 1))
    # x = torch.zeros((num, seq_len, 5))
    # y = torch.zeros((num, 1))
    cur_seq_len = 0
    cur_num = 0

    with open(input_path, "rb") as infile:
        with open(target_path, "rb") as tarfile:
            cur_seq_len = 0
            cur_num = 0
            tar_num = 0
            for tarline in tarfile:
                tar_num+=1
                tarline = tarline.rstrip().split("\t")
                if(len(tarline) != 4):
                    continue
                for inline in infile:
                    inline = inline.rstrip().split("\t")
                    if(len(inline) != 4):
                        continue
                    
                    #cnv_in_seq = 'No'
                    tar_chrom = tarline[0] if 'chr' in tarline[0] else 'chr' + tarline[0]
                    tar_start = int(tarline[1])
                    tar_end = int(tarline[2])
                    tar_cnv = tarline[3]

                    in_chrom = inline[0]
                    in_pos = int(inline[1])
                    in_base = inline[2].upper()
                    in_read_depth = inline[3]
                     
                    # Check if chromosomes are equal
                    if(tar_chrom == in_chrom):
                        if(tar_start <= in_pos and in_pos <= tar_end):
                            # Check if in_base is a valid base (A, T, C or G)
                            if(in_base in all_bases): 
                                seq_tensor = seq_to_tensor(in_base)
                            else:
                                continue
                            rd_tensor = np.array([[[int(in_read_depth)]]])
                            x[cur_num][cur_seq_len] = np.concatenate((seq_tensor, rd_tensor),2)
                            # Sets cnv_in_seq to 'Yes' if there is a CNV between start and end
                            if(tar_cnv == 'Yes'):
                                cnv_in_seq = 'Yes'
                            else:
                                cnv_in_seq='No'
                            
                            cur_seq_len+=1

                            # When cur_seq_len is equal to seq_len set y and increment cur_num
                            if(cur_seq_len == seq_len):

                                y[cur_num] = target_to_one_hot(cnv_in_seq)
                                cur_seq_len = 0
                                cur_num+=1
                                cnv_in_seq = False
                                # if current number of training examples is equal to total number of training examples
                                # then return
                                if(cur_num == num):
                                    print("Found %d training examples." %(num))
                                    return x, y

                        # pos in input file has passed the end pos in the target file                      
                        elif(in_pos > tar_end):
                            break
                        
                        # pos in input file has not reached the start pos in the target file
                        elif(in_pos < tar_start):
                            continue

                    # go to next inline if chromosomes are not equal
                    else:
                        continue
    print("Did not load enough training examples")
    return x, y
    
def load_data(input_path, target_path, num, seq_len, use_batch=False):
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
   
    print("Loading all the data...")
    x, y = load(num, input_path, target_path, seq_len)
         
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
    input_out_file = np.save(in_out_path, input_data)
    target_out_file = np.save(tar_out_path, target_data)

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
    x = np.load(in_path)
    y = np.load(tar_path)

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

class Corpus(object):
    # def __init__(self, path):
    def __init__(self, train_in_path, train_tar_path, val_in_path, val_tar_path, test_in_path, test_tar_path, num, seq_len):
        # self.dictionary = Dictionary()
        # self.train = self.tokenize(os.path.join(path, 'train.txt'))
        # self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))
        train_in_txt = "data/input_train.out"
        train_tar_txt = "data/target_train.out"

        val_in_txt = "data/input_val.out"
        val_tar_txt = "data/target_val.out"

        test_in_txt = "data/input_test.out"
        test_tar_txt = "data/target_test.out"

        train_in_path = "data/train_in.npy"
        train_tar_path = "data/train_tar.npy"
        val_in_path = "data/val_in.npy"
        val_tar_path = "data/val_tar.npy"
        test_in_path = "data/test_in.npy"
        test_tar_path = "data/test_tar.npy"

        try:
            train_x, train_y = load_data_from_file(train_in_path, train_tar_path)
            # train_x = np.load('data/fake_in.npy')
            # train_y = np.load('data/fake_tar.npy')
        except:
            print("Could not load presaved training data")
            train_x, train_y = load_data(train_in_txt, train_tar_txt, num, seq_len)
            save_data(train_x, train_y, train_in_path, train_tar_path)
        
        try:
            val_x, val_y = load_data_from_file(val_in_path, val_tar_path)
            # val_x = np.load('data/fake_in.npy')
            # val_y = np.load('data/fake_tar.npy')
        except:
            print("Could not load presaved validation data")
            val_x, val_y = load_data(val_in_txt, val_tar_txt, num, seq_len)
            save_data(val_x, val_y, val_in_path, val_tar_path)

        try:
            test_x, test_y = load_data_from_file(test_in_path, test_tar_path)
            # test_x = np.load('data/fake_in.npy')
            # test_y = np.load('data/fake_tar.npy')
        except:
            print("Could not load presaved test data")
            test_x, test_y = load_data(test_in_txt, test_tar_txt, num, seq_len)
            save_data(test_x,test_y, test_in_path, test_tar_path)

        # print(torch.from_numpy(x).shape)
        train_in = torch.from_numpy(train_x).float()
        train_tar = torch.from_numpy(train_y).float()

        val_in = torch.from_numpy(val_x).float()
        val_tar = torch.from_numpy(val_y).float()

        test_in = torch.from_numpy(test_x).float()
        test_tar = torch.from_numpy(test_y).float()
    
        #import pdb; pdb.set_trace()
        # Flatten data
        self.train_in = train_in.view(train_in.shape[0], -1)
        self.train_tar = train_tar.view(train_tar.shape[0], -1)

        self.val_in = val_in.view(val_in.shape[0], -1)
        self.val_tar = val_tar.view(val_tar.shape[0], -1)

        self.test_in = test_in.view(test_in.shape[0], -1)
        self.test_tar = test_tar.view(test_tar.shape[0], -1)
    # def tokenize(self, path):
    #     """Tokenizes a text file."""
    #     assert os.path.exists(path)
    #     # Add words to the dictionary
    #     with open(path, 'r') as f:
    #         tokens = 0
    #         for line in f:
                

    #     # Tokenize file content
    #     with open(path, 'r') as f:
    #         ids = torch.LongTensor(tokens)
    #         token = 0
    #         for line in f:
    #             words = line.split() + ['<eos>']
    #             for word in words:
    #                 ids[token] = self.dictionary.word2idx[word]
    #                 token += 1

    #     return ids
