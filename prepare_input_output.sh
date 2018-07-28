#!/bin/bash

dir=data

while IFS=, read input_path target_path;do
    submitjob python ../clean_data_single.py --input_data $input_path --target_data $target_path --output_dir clean
done < data_list.txt

