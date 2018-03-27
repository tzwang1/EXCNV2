import numpy as np

total = 1000000

x = np.zeros((total, 5))
y = np.zeros((total, 1))

for i in range(0, total):
    x_value = np.random.randint(0,4)
    depth = np.random.randint(5, 15)
    y_value = np.random.randint(0, 2)

    x[i][x_value] = 1
    x[i][4] = depth
    y[i] = y_value

np.save('data/fake_in.npy', x)
np.save('data/fake_tar.npy', y)
    