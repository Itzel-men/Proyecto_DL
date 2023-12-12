import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

from DAST_Network import *
from DAST_utils import *

import scipy.io as sio
import time

from torch import nn, optim
from torch.autograd import Variable

#Myscore function
def myScore(Target, Pred):
    tmp1 = 0
    tmp2 = 0
    for i in range(len(Target)):
        if Target[i] > Pred[i]:
            tmp1 = tmp1 + math.exp((-Pred[i] + Target[i]) / 13.0) - 1
        else:
            tmp2 = tmp2 + math.exp((Pred[i] - Target[i]) / 10.0) - 1
    tmp = tmp1 + tmp2
    return tmp

#Hiperparámetros
batch_size = 256
lr = 1e-2
n_epochs = 3
dim_val_s= 64
dim_attn_s= 64

dim_val_t = 64
dim_attn_t = 64

dim_val = 64
dim_attn = 64

time_step= 32
input_size = 2

dec_seq_len = 4
out_seq_len = 1

n_encoder_layers = 2
n_decoder_layers = 1

dropout = 0.2
n_heads = 4

testX1_4 = sio.loadmat('Xtest_new_bearing1_4.mat')  #upload sliding time window processed data
testX1_4 = testX1_4['testX1_4_new']

testY1_4 = sio.loadmat('Ytest_bearing1_4.mat')
testY1_4 = testY1_4['Ytest_bearing1_4']

X_test = Variable(torch.Tensor(testX1_4).float())
Y_test = Variable(torch.Tensor(testY1_4.T).float())

test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

args={}
kwargs={}

args['batch_size'] = batch_size
args['test_batch_size'] = batch_size

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = args['test_batch_size'],
                                          shuffle=True)

#Modelo DAST
model_bearing = DAST(dim_val_s=dim_val_s, dim_attn_s= dim_attn_s, dim_val_t=dim_val_t, dim_attn_t=dim_attn_t,
             dim_val = dim_val, dim_attn= dim_attn, time_step = time_step, input_size = input_size,dec_seq_len=dec_seq_len,
             out_seq_len=out_seq_len, n_decoder_layers=n_decoder_layers, n_encoder_layers=n_encoder_layers,
             n_heads=n_heads, dropout=dropout)

optimizer = optim.Adam(model_bearing.parameters(), lr=lr)
criterion = nn.MSELoss()

model_bearing = torch.load('B1_DAST_prediciton_model')

Y_test_numpy = Y_test.detach().numpy()
test_list = []

for k ,(batch_x,batch_y) in enumerate(test_loader):
    prediction = model_bearing(batch_x)
    prediction[prediction<0] = 0
    test_list.append(prediction)

np.savetxt("prediction_B4.txt", prediction)

test_all =  torch.cat(test_list).detach().numpy()
test_all_tensor = torch.from_numpy(test_all)
test_loss = torch.sqrt(criterion(test_all_tensor*125, Y_test*125))
test_score = myScore(Y_test_numpy*125, test_all*125)

np.savetxt("prediction_losses_B4.txt", test_loss.item())
