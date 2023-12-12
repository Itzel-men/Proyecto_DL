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

if __name__ == '__main__':
    #Datos preprocesados
    trainX1_2_new = sio.loadmat('Xtrain_new_bearing1_2.mat')  #Datos bel bearing2
    trainX1_2_new = trainX1_2_new['trainX1_2_new']
    trainX1_1_new = sio.loadmat('Xtrain_new_bearing1_1.mat')  #Datos del bearing1
    trainX1_1_new = trainX1_1_new['trainX1_1_new']

    #Etiquetas correspondientes a cada bearing
    trainY1_2_new = sio.loadmat('Ytrain_bearing1_2.mat')
    trainY1_2_new = trainY1_2_new['Ytrain_bearing1_2']
    trainY1_1_new = sio.loadmat('Ytrain_bearing1_1.mat')
    trainY1_1_new = trainY1_1_new['Ytrain_bearing1_1']

    #Conjunto de entrenamiento
    trainX_ = np.concatenate((trainX1_1_new, trainX1_2_new), axis=0)

    #Etiquetas del conjunto de entrenamiento
    trainY_ = np.concatenate((trainY1_1_new, trainY1_2_new), axis=1)

    
    X_train = Variable(torch.Tensor(trainX_).float())
    Y_train = Variable(torch.Tensor(trainY_.T).float())

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

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    #Train loader
    args={}
    kwargs={}

    args['batch_size'] = batch_size
    args['test_batch_size'] = batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = args['batch_size'],
                                           shuffle=True)

    #Modelo DAST
    model_bearing = DAST(dim_val_s=dim_val_s, dim_attn_s= dim_attn_s, dim_val_t=dim_val_t, dim_attn_t=dim_attn_t,
             dim_val = dim_val, dim_attn= dim_attn, time_step = time_step, input_size = input_size,dec_seq_len=dec_seq_len,
             out_seq_len=out_seq_len, n_decoder_layers=n_decoder_layers, n_encoder_layers=n_encoder_layers,
             n_heads=n_heads, dropout=dropout)

    optimizer = optim.Adam(model_bearing.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    val_losses = []

    loss_values = []
    val_losses = []

    # Training loop
    for epoch in range(n_epochs):
        # Uses loader to fetch one mini-batch for training
        train_loss = 0.
        model_bearing.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.float()
            y_batch = y_batch.float()
            yhat = model_bearing(x_batch)

            #loss_fn.requires_grad = True
            loss_ = torch.sqrt(criterion(yhat,y_batch))
            loss_.backward()
            optimizer.step()

            train_loss += loss_.item()

        losses = train_loss/len(train_loader)
        loss_values.append(losses)
        #print(f'Época [{epoch+1}/{n_epochs}], Loss: {losses:.4f}')

        #Model save
        File_Path = '..' + '\\' + 'DAST' + '\\' + 'B1' + '\\' 
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)
        torch.save(model_bearing, File_Path + '/' + 'B1_DAST_prediciton_model')