import numpy as np

total = 1000000
seq_len = 30

x = np.zeros((total, seq_len, 5))
y = np.zeros((total, 1))

for i in range(0, total):
    for j in range(0, seq_len):
        x_value = np.random.randint(0,4)
        depth = np.random.randint(5, 15)
        x[i][j][x_value] = 1
        x[i][j][4] = depth
    
    y_value = np.random.randint(0, 2)

    y[i] = y_value

np.save('data/fake_in.npy', x)
np.save('data/fake_tar.npy', y)
    