import pandas as pd

input_path = 
output_path = 

x = pd.read_csv(input_path, sep='\t')
x[x.columns[0]] = x[x.columns[0]].map(lambda x: x.lstrip('chr'))
x[x.columns[2]] = x = x[x.columns[2]].str.upper()

x.to_csv(output_path, sep='\t', index=False)