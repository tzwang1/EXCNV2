import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import subprocess
import argparse
import pandas as pd

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

# TODO: Currently this function is always getting batches starting from the start of the file.
# Need to either load the whole file into memory (could use too much memory), or allow the function
# to randomly access the file to start reading from another index

# Temporary work around, use a batch size equal to the number of training examples
def load_batch(input_path, target_path, batch_size, seq_len):
    x_batch = torch.zeros((batch_size, seq_len, 5))
    y_batch = torch.zeros((batch_size, 1))
    with open(input_path, "rb") as infile:
        with open(target_path, "rb") as tarfile:
            cur_seq_len = 0
            cur_batch_size = 0
            cnv_in_seq = False        
            
            for inline in infile:
                inline = inline.rstrip().split("\t")
                for tarline in tarfile:
                    tarline = tarline.rstrip().split("\t")
                    if(len(tarline) == 4):
                        tar_chrom = tarline[0] if 'chr' in tarline[0] else 'chr'+tarline[0]
                        tar_start = tarline[1]
                        tar_end = tarline[2]
                        tar_cnv = tarline[3]

                        in_chrom = inline[0]
                        in_pos = inline[1]
                        in_base = inline[2].upper()
                        in_read_depth = inline[3]

                        if(tar_chrom == in_chrom):
                            if(tar_start < in_pos and in_pos < tar_end):
                                seq_tensor = seq_to_tensor(in_base)
                                rd_tensor = torch.FloatTensor([[[int(in_read_depth)]]])

                                x_batch[cur_batch_size][cur_seq_len] = torch.cat((seq_tensor, rd_tensor),2)

                                # Sets cnv_in_seq to True if there is a CNV between start and end
                                if(tar_cnv == 'Yes'):
                                    cnv_in_seq = True
                                
                                cur_seq_len+=1

                                # When cur_seq_len is equal to seq_len start preparing a new batch
                                if(cur_seq_len == seq_len):
                                    cur_seq_len = 0
                                    y_batch[cur_batch_size] = target_to_one_hot(cnv_in_seq)
                                    cur_batch_size+=1
                                    if(cur_batch_size == batch_size):
                                        return x_batch, y_batch
                                # After adding data for current line in input file break the inner loop to go to the next input line
                                break
                            elif(in_pos > tar_end):
                                break

                    else:
                        continue

    print("DID NOT LOAD ENOUGH SAMPLES")     
    return x_batch, y_batch

# Helper method for load_data
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
                    
                    cnv_in_seq = 'No'
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
                            # print("Found valid pos")
                            # print(tar_start)
                            # print(tar_end)
                            # print(in_pos)
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
                            
                            cur_seq_len+=1

                            # When cur_seq_len is equal to seq_len set y and increment cur_num
                            if(cur_seq_len == seq_len):

                                #if(cnv_in_seq == 'No'):
                                #    print("cnv_in_seq = No")

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
    return x, y
    
def load_data(input_path, target_path, num, batch_size, seq_len, use_batch=False):
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
    if not(use_batch):
        print("Not using batch")
        x, y = load(num, input_path, target_path, seq_len)
 
    else:
        x = np.zeros((num, batch_size, seq_len, 5))
        y = np.zeros((num, 1))

        for i in range(num):
            x[i], y[i] = load_batch(input_path, target_path, batch_size, seq_len)

        #x, y = load_batch(input_path, target_path, num, seq_len)
         
    return x, y


def save_data(input_data, target_data, in_out_path="data/input_data.npy", tar_out_path="data/target_data.npy"):
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

def load_data_from_file(in_path="data/input_data.npy", tar_path="data/target_data.npy"):
    '''
    Args:
        in_path: path to input file
        tar_path: path to target file
    
    Returns:
        x: numpy array of input data
        y: numpy array of target data
    '''
    x = np.load(in_path)
    y = np.load(tar_path)

    return x, y

# Generates input data
def generate_data(bamfile, ref_genome, outfile):
    '''
    Args:
        bamfile: bamfile to analyze
        bedfile: bedfile associated with bamfile
        ref_genome: reference genome
        outfile: output file

    Saves file:
      input file with four tab delimited columns:
      1. Chromosome number
      2. Start
      3. Base
      4. Read depth
    '''
    SAMTOOLS = "samtools mpileup -q 15 -Q 20 -f " + ref_genome + " -s " +bamfile+ " | cut -f 1,2,3,4 > " + outfile
    subprocess.call(SAMTOOLS, shell=True)


# Generates target file from CNV file download from TCGA
def generate_target_data(input_path, target_path):
    '''
    Args:
        input_path = input file with CNV information
        target_path = target file
    Saves file:
        target file with four tab delimited columns:
        1. Chromosome number
        2. start point
        3. end point
        4. Yes/No CNV found
    '''
    target = pd.read_csv(input_path, delimiter='\t')
    target = target.drop(target.columns[[0,4]], axis=1)
    mean_column = target.columns[3]
    target.loc[target[mean_column] > 0.2, mean_column] = 'gain'
    target.loc[target[mean_column] < -0.2, mean_column] = 'loss'
    target.loc[(target[mean_column] > -0.2) & (target[mean_column] < 0.2), mean_column] = 'neutral'
    target.to_csv(target_path, sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument('--bam', type=str, help='location of the bam file')
    parser.add_argument('--ref', type=str, default='~/truman/data/hg38.fasta', help='location of reference file')
    parser.add_argument('--in_out', type=str, help='location of output for input file')
    parser.add_argument('--cnv', type=str, help='location of cnv file')
    parser.add_argument('--tar_out', type=str, help='location of output for target file')
  
    args = parser.parse_args()

    bamfile = args.bam
    ref_genome = args.ref
    outfile = args.in_out
    
    cnvfile = args.cnv
    target_outfile = args.tar_out
    
    print("Running generate data")
    generate_data(bamfile, ref_genome, outfile)
    print("Running generate target data")
    generate_target_data(cnvfile, target_outfile)
