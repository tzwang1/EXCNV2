import pandas as pd

def clean(input_path, output_path):
    x = pd.read_csv(input_path, sep='\t', header=None)
    x[x.columns[0]] = x[x.columns[0]].map(lambda x: x.lstrip('chr'))
    x[x.columns[2]] = x[x.columns[2]].str.upper()
    # x[x.columns[3]] = x[x.columns[3]].astype(int)

    x.to_csv(output_path, sep='\t', index=False)


input_train = "data/good_data/input_train.out" 
output_train = "data/good_data/input_train2.out"

input_val = "data/good_data/input_val.out" 
output_val = "data/good_data/input_val2.out"

input_test = "data/good_data/input_test.out" 
output_test = "data/good_data/input_test2.out"

clean(input_train, output_train)
clean(input_val, output_val)
clean(input_test, output_test)


