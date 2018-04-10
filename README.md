# PyTorch RNN/LSTM for CNV Detection

This project was to train a multi-layer RNN (Elman, GRU, or LSTM) on a CNV detection task.
By default, the training script uses the data files (data_x.pl and data_y.pl), provided.
The trained model can then be used to detect CNVs in other sequences

```bash
python main.py --lr 0.0005 --epoch 30      # Train a LSTM on data_x.pl and data_y.pl for 30 epochs reaching training accuracy of 95%, validation accuracy of 67% and test accuracy of 61%
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data folder
  --data_in          name of the input data
  --data_tar         name of the target data
  --fcout1           size of the output for the first fully connected layer
  --fcout2           size of the output for the second fully connected layer
  --decode1          size of the output for the third fully connected layer
  --show_every       numbero of epochs to train on before showing accuracy of training and validation set 
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

With these arguments, a variety of models can be tested.


