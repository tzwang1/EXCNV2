import numpy as np
from data import load_data_from_file
from data import load_data
from data import save_data

# Reduce size 10000000 data to include both 1 and 0
def reduce_data(input_data, target_data):
    input_data = input_data[6000000:7000000]
    target_data = target_data[6000000:7000000]

    return input_data, target_data

num = 10000000

test_in_txt = "data/input_test.out"
test_tar_txt = "data/target_test.out"

train_in_path = "data/train_in.npy"
train_tar_path = "data/train_tar.npy"
val_in_path = "data/val_in.npy"
val_tar_path = "data/val_tar.npy"
test_in_path = "data/test_in.npy"
test_tar_path = "data/test_tar.npy"

test_x, test_y = load_data(test_in_txt, test_tar_txt, num, 30)
save_data(test_x,test_y, test_in_path, test_tar_path)


# train_x, train_y = load_data_from_file(train_in_path, train_tar_path)
# val_x, val_y = load_data_from_file(val_in_path, val_tar_path)
# test_x, test_y = load_data_from_file(test_in_path, test_tar_path)

# print(train_x.shape)
# print(val_x.shape)
# print(test_x.shape)

# train_x, train_y = reduce_data(train_x, train_y)
# val_x, val_y = reduce_data(val_x, val_y)
# test_x, test_y = reduce_data(test_x, test_y)

# print(train_x.shape)
# print(val_x.shape)
# print(test_x.shape)

# print(np.sum(train_x))
# print(np.sum(val_x))
# print(np.sum(test_x))

# print(np.sum(train_y))
# print(np.sum(val_y))
# print(np.sum(test_y))

# np.save('new_data/train_in.npy', train_x)
# np.save('new_data/train_tar.npy', train_y)

# np.save('new_data/val_in.npy', val_x)
# np.save('new_data/val_tar.npy', val_y)

# np.save('new_data/test_in.npy', test_x)
# np.save('new_data/test_tar.npy', test_y)



