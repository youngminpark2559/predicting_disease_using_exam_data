# conda activate py36gputorch100 && \
# cd /mnt/external_disk/Companies/B_Link/Project_code/DeepLearning_predicting_disease/Codes/Src && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import sys 
sys.path.append('..')
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

import utils as utils

torch.manual_seed(1)    # reproducible

# ================================================================================
BATCH_SIZE=8000
# BATCH_SIZE=1000
# BATCH_SIZE = 8
number_of_features=4

# ================================================================================
class Net(torch.nn.Module):
  def __init__(self,n_feature,n_hidden,n_output):
    super(Net,self).__init__()
    self.hidden=torch.nn.Linear(n_feature,n_hidden)   # hidden layer
    self.hidden2=torch.nn.Linear(n_hidden,n_hidden)   # hidden layer
    self.predict=torch.nn.Linear(n_hidden,n_output)   # output layer
    self.dropout=torch.nn.Dropout(p=0.7)

  def forward(self,x):
    after_hidden=self.hidden(x)
    after_hidden=F.relu(after_hidden)
    # after_hidden=self.dropout(after_hidden)

    # after_hidden2=self.hidden2(after_hidden)
    # after_hidden2=F.relu(after_hidden2)
    # after_hidden2=self.dropout(after_hidden2)

    x=self.predict(after_hidden)
    
    return x

class Medical_Dataset(data.Dataset):
  def __init__(self):

    loaded_csv=utils.load_csv('../../Data/Train/NHID2013.csv')
    # print("loaded_csv",loaded_csv)

    # print("loaded_csv",loaded_csv.shape)
    # (3081208, 6)
    min_max_scaler=preprocessing.MinMaxScaler()
    loaded_csv_minmax=pd.DataFrame(min_max_scaler.fit_transform(loaded_csv.iloc[:,:4]))
    # print(loaded_csv_minmax)
    # print("loaded_csv_minmax",loaded_csv_minmax.shape)
    # loaded_csv_minmax (234426, 6)

    
    loaded_csv_minmax=pd.concat([loaded_csv_minmax,loaded_csv.iloc[:,4:]],axis=1)
    # print("loaded_csv_minmax",loaded_csv_minmax)

    self.train_X=np.array(loaded_csv_minmax.iloc[:3000000,:4])
    self.train_y=np.array(loaded_csv_minmax.iloc[:3000000,4:])
    # self.train_X=np.array(loaded_csv_minmax.iloc[:3000,:4])
    # self.train_y=np.array(loaded_csv_minmax.iloc[:3000,4:])
    # print("train_X",train_X.shape)
    # print("train_y",train_y.shape)

    self.number_of_data=self.train_X.shape[0]

  # ================================================================================
  def __len__(self):
    return self.number_of_data

  # ================================================================================
  def __getitem__(self,idx):
    return self.train_X[idx],self.train_y[idx]

class Medical_Test_Dataset(data.Dataset):
  def __init__(self):

    loaded_csv=utils.load_csv('../../Data/Train/NHID2013.csv')
    # print("loaded_csv",loaded_csv)

    min_max_scaler=preprocessing.MinMaxScaler()
    loaded_csv_minmax=pd.DataFrame(min_max_scaler.fit_transform(loaded_csv.iloc[:,:4]))
    # print(loaded_csv_minmax)
    # print("loaded_csv_minmax",loaded_csv_minmax.shape)
    # loaded_csv_minmax (234426, 6)

    loaded_csv_minmax=pd.concat([loaded_csv_minmax,loaded_csv.iloc[:,4:]],axis=1)

    self.test_X=np.array(loaded_csv_minmax.iloc[3000000:,:4])
    self.test_y=np.array(loaded_csv_minmax.iloc[3000000:,4:])
    # print("train_X",train_X.shape)
    # print("train_y",train_y.shape)

    self.number_of_data=self.test_X.shape[0]

  # ================================================================================
  def __len__(self):
    return self.number_of_data

  # ================================================================================
  def __getitem__(self,idx):
    return self.test_X[idx],self.test_y[idx]

# ================================================================================
net=Net(n_feature=number_of_features,n_hidden=100,n_output=2).cuda()     # define the network

optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
# loss_func=FocalLoss(class_num=15)
loss_func=torch.nn.MSELoss()  # this is for regression mean squared loss

torch_dataset=Medical_Dataset()
loader=data.DataLoader(
  dataset=torch_dataset,      # torch TensorDataset format
  batch_size=BATCH_SIZE,      # mini batch size
  shuffle=True,               # random shuffle for training
  num_workers=2,              # subprocesses for loading data
)

torch_test_dataset=Medical_Test_Dataset()
test_loader=data.DataLoader(
  dataset=torch_test_dataset,      # torch TensorDataset format
  batch_size=BATCH_SIZE,      # mini batch size
  shuffle=True,               # random shuffle for training
  num_workers=2,              # subprocesses for loading data
)

# ================================================================================
# def show_batch():
#   loss_list=[]
#   for epoch in range(30):   # train entire dataset 3 times
#     for step,(batch_x,batch_y) in enumerate(loader):  # for each training step
#       batch_x=batch_x.float().cuda()
#       # print("batch_x",batch_x.shape)
#       batch_x=batch_x.view(BATCH_SIZE,number_of_features)
#       # print("batch_x",batch_x.shape)
#       # batch_x torch.Size([20, 1])
#       batch_y=batch_y.float().cuda()
#       # print("batch_y",batch_y.shape)
#       # batch_y torch.Size([5, 2])
     
#       prediction=net(batch_x)     # input x and predict based on x
#       # print("prediction",prediction.shape)
#       # print("batch_y",batch_y.shape)
      
#       # prediction=prediction.view(5,4)

#       loss=loss_func(prediction,batch_y)     # must be (1. nn output, 2. target)
#       print("loss",loss.item())
#       loss_list.append(loss.item())

#       optimizer.zero_grad()   # clear gradients for next train
#       loss.backward()         # backpropagation, compute gradients
#       optimizer.step()        # apply gradients

#       # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy())
  
#   from matplotlib import pyplot as plt
#   plt.plot(loss_list)
#   plt.show()
#   # /mnt/external_disk/Capture_temp/2020_04_25_20:44:25.png

# ================================================================================
def train_by_batches():
  net.train()
  loss_list=[]
  for epoch in range(50):   # train entire dataset 3 times
  # for epoch in range(3):   # train entire dataset 3 times
    for step,(batch_x,batch_y) in enumerate(loader):  # for each training step
      batch_x=batch_x.float().cuda()
      # print("batch_x",batch_x.shape)
      batch_x=batch_x.view(BATCH_SIZE,number_of_features)
      # print("batch_x",batch_x.shape)
      # batch_x torch.Size([20, 1])
      batch_y=batch_y.float().cuda()
      # print("batch_y",batch_y.shape)
      # batch_y torch.Size([5, 2])
     
      prediction=net(batch_x)     # input x and predict based on x
      # print("prediction",prediction.shape)
      # print("batch_y",batch_y.shape)
      
      # prediction=prediction.view(5,4)

      loss=loss_func(prediction,batch_y)     # must be (1. nn output, 2. target)
      print("loss",loss.item())
      loss_list.append(loss.item())

      optimizer.zero_grad()   # clear gradients for next train
      loss.backward()         # backpropagation, compute gradients
      optimizer.step()        # apply gradients

      # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy())
  
  plt.plot(loss_list)
  plt.show()
  # /mnt/external_disk/Capture_temp/2020_04_25_20:44:25.png

# ================================================================================
def test_by_batches():
  with torch.no_grad():
    net.eval()

    for step,(batch_x,batch_y) in enumerate(test_loader):  # for each training step
      batch_x=batch_x.float().cuda()
      # print("batch_x",batch_x.shape)
      batch_x=batch_x.view(BATCH_SIZE,number_of_features)
      # print("batch_x",batch_x.shape)
      # batch_x torch.Size([20, 1])
      batch_y=batch_y.float().cuda()
      # print("batch_y",batch_y.shape)
      # batch_y torch.Size([5, 2])
      
      prediction=net(batch_x)     # input x and predict based on x
      # print("prediction",prediction.shape)
      # print("batch_y",batch_y.shape)
      # print("prediction",prediction)
      # print("batch_y",batch_y)
      pred_np=np.array(prediction.detach().cpu())
      # print("pred_np",pred_np.shape)
      target_np=np.array(batch_y.detach().cpu())

      pred_np=np.round_(pred_np,decimals=0,out=None)
      pred_np=np.clip(pred_np,1,15,out=None)
      # print("target_np.shape[0]",target_np.shape[0])
      target_np_flat=target_np.reshape(-1)
      print("target_np_flat",target_np_flat.shape)
      pred_np_flat=pred_np.reshape(-1)
      print("pred_np_flat",pred_np_flat.shape)
      print((target_np_flat==pred_np_flat).sum())
      print((target_np_flat==pred_np_flat).sum()/target_np_flat.shape[0])

# 8000
# 7251
# 0.906375
# 8000
# 7264
# 0.908
# 8000
# 7250
# 0.90625
# 8000
# 7263
# 0.907875
# 8000
# 7195
# 0.899375
# 8000
# 7203
# 0.900375
# 8000
# 7195
# 0.899375
# 8000
# 7212
# 0.9015
# 8000
# 7259
# 0.907375
# 8000
# 7278
# 0.90975

# ================================================================================
train_by_batches()
test_by_batches()