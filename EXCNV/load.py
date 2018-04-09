import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import subprocess
import argparse
import pandas as pd

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
    parser.add_argument('--ref', type=str, help='location of reference file')
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
