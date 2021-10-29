import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions import  normal
import math
import setproctitle
from torch import optim

torch.manual_seed(0)
np.random.seed(0)
n_epoch=4000

dim_z=30
que=40
train=True
p="region_features_NYC.csv"
with open(p,encoding = 'utf-8') as f:
    inputdata_tr = np.loadtxt(f,delimiter = ",", skiprows = 1)
    inputdata_tr = inputdata_tr[:, 1:]
inputdata_1=inputdata_tr
if train:
    devdata = inputdata_tr[-650:]
    batch_size_dev = 650
    batch_size = 2167-batch_size_dev
    inputdata_train = inputdata_tr[:batch_size]
else:
    inputdata = np.load("ft.npy")
    batch_size = 216

# inputdata=np.load("train_65_new.npy")
# batch_size = 1952
city="Sea"
p="region_features_"+city+".csv"
with open(p,encoding = 'utf-8') as f:
    inputdata_test = np.loadtxt(f,delimiter = ",", skiprows = 1)
    inputdata_test = inputdata_test[:, 1:]
all=np.concatenate((inputdata_tr,inputdata_test),axis=0)
batch_size_all=len(inputdata_train)
batch_size_test=len(inputdata_test)
max_ = np.max(all, axis = 0)
min_ = np.min(all, axis = 0)
inputdata_train = (inputdata_train - min_) * 2 / (max_ - min_) -1
inputdata_test = (inputdata_test - min_) * 2 / (max_ - min_) -1
devdata= (devdata - min_) * 2 / (max_ - min_) -1
data_quan=np.concatenate((inputdata_train,devdata,inputdata_test),axis=0)
mean_=np.mean(data_quan, axis = 0)
inputdata_train=inputdata_train-mean_
inputdata_test=inputdata_test-mean_
data_quan_all=data_quan-mean_
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_accessible(A):
	B = A
	item = A
	for i in range(len(A)):
		item = np.matmul(item,A)
		B = B + item
	return B
data=np.loadtxt("best_graph_DAWG_NYC.txt")
class p_z_(nn.Module):

    def __init__(self, dim_in=20, dim_out_con=6):
        super().__init__()
        self.input = nn.Linear(dim_in, 10)
        # dim_in is dim of latent space z
        self.output = nn.Linear(10, dim_out_con)


    def forward(self, z_input,sig1=False,sig=False):
        z = F.elu(self.input(z_input))
        z_out=F.elu(self.output(z))


        return z_out

class p_z_1(nn.Module):

    def __init__(self, dim_in=20, dim_out_con=6):
        super().__init__()
        self.input = nn.Linear(dim_in, 10)
        # dim_in is dim of latent space z
        self.output = nn.Linear(10, dim_out_con)


    def forward(self, z_input,sig1=False,sig=False):
        z = F.elu(self.input(z_input))
        z_out=F.tanh(self.output(z))


        return z_out

global data1
data1=list(data)

data_sum0=np.sum(data,axis=0)
node_father=np.where(data_sum0==0)



data_sum1=np.sum(data,axis=1)
node_child=np.where(data_sum1==0)
criterion = nn.MSELoss()



data_remove=[]
global data_all
data_all=list(np.arange(len(data)))
for inter1 in node_child[0]:
    data_all[inter1]=-1
    data_remove.append(inter1)
    data1[inter1]=np.ones((60))
while data_all!=[-1]*60:
    for i in range(len(data_all)):
        index_s=np.where(data1[i]==1)
        index_s=list(index_s[0])
        commonEle = [val for val in index_s if val in data_all]
        if len(commonEle)==0 and data_all[i]!=-1:
            data_remove.append(data_all[i])
            data_all[i]=-1
            data1[i] = np.ones((60))
            break

data_ch_father=[]
for inter in data_remove:
    if data[:, inter].sum()==0:
        data_ch_father.append(inter)
for inter1 in data_ch_father:
    data_remove.remove(inter1)
data_remove=data_remove+data_ch_father


data_re=np.array(data_remove)[:que][::-1]
data_save=np.array(data_remove)[que:]



data_dec=np.array(data_remove)[::-1]
#

params=[]

dict_re_func_dec=p_z_1(int(dim_z), int(len(data_re))).to(device)

params=params+list(dict_re_func_dec.parameters())



loss_test_all=[]
p_o_z = p_z_(int(len(data_save)), dim_z).to(device)
params=params+list(p_o_z.parameters())

optimizer = optim.Adam(params, lr=8e-5)
loss_max=-10000
loss_train=[]
loss_dev=[]
loss_test=[]
loss_test_mse=[]
data_last_test=torch.FloatTensor(inputdata_test).to(device)
data_quan=torch.FloatTensor(inputdata_train).to(device)
data_last_dev = torch.FloatTensor(devdata).to(device)
data_quan_all=torch.FloatTensor(data_quan_all).to(device)

for epoch in range(n_epoch):
    if train:
        # ### dec
        # p(z)
        z_infer = p_o_z(data_quan[:,data_save])
        z_infer_1 = p_o_z(data_quan_all[:, data_save])

        #

        # RECONSTRUCTION LOSS

        data_now = z_infer
        z_train=z_infer_1[:2167]
        dict_re_dis_dec = dict_re_func_dec(data_now)
        loss_mean=criterion(dict_re_dis_dec,data_quan[:,list(data_re)])




        print("loss_RECONSTRUCTION:", loss_mean)
        # print("loss_REGULARIZATION:", torch.mean(loss_REGULARIZATION))
        # # print("loss_AUXILIARY:", torch.mean(np.sum(loss_AUXILIARY,axis=0)))
        # print("loss:",loss_mean)
        objective = loss_mean
        optimizer.zero_grad()
        # Calculate gradients
        objective.backward()
        # Update step
        optimizer.step()

        # dev_loss
        data_inz_all = data_last_dev[:, data_save]
        z_infer_sample = p_o_z(data_inz_all)
        p_z_test = z_infer_sample
        data_now = z_infer_sample
        dict_re_dis_dec = dict_re_func_dec(data_now)
        loss_RECONSTRUCTION_dev =criterion(dict_re_dis_dec,data_last_dev[:,list(data_re)])

        loss_cal=-loss_RECONSTRUCTION_dev
        print("loss_dev:",loss_cal)


        # test_loss
        data_inz_all = data_last_test[:, data_save]
        z_infer_sample = p_o_z(data_inz_all)
        # p_z_test = z_infer_sample
        data_now = z_infer_sample
        y_hat=dict_re_func_dec(data_now)
        # loss_RECONSTRUCTION_test_mse_re=criterion(y_hat[:,list(data_re)],data_last_test[:,list(data_re)])
        # loss_RECONSTRUCTION_test_mse_save=criterion(y_hat[:,list(data_save)],data_last_test[:,list(data_save)])
        loss_RECONSTRUCTION_test_mse=(y_hat-data_last_test[:,list(data_re)])**2
        loss_RECONSTRUCTION_test_mse=torch.sum(loss_RECONSTRUCTION_test_mse,1)
        loss_RECONSTRUCTION_test_mse=torch.mean(loss_RECONSTRUCTION_test_mse,0)
        # loss_train.append(float(torch.mean(loss_RECONSTRUCTION).cpu().data.numpy()))
        loss_dev.append(float(torch.mean(loss_RECONSTRUCTION_dev).cpu().data.numpy()))
        # # loss_test.append(float(torch.mean(loss_RECONSTRUCTION_test).cpu().data.numpy()))
        loss_test_mse.append(float(loss_RECONSTRUCTION_test_mse.cpu().data.numpy()))
        # print("loss_RECONSTRUCTION_test:", torch.mean(loss_RECONSTRUCTION_test))
        print("loss_RECONSTRUCTION_test_mse:", loss_RECONSTRUCTION_test_mse)
        if loss_cal>loss_max:
            loss_max=loss_cal
            z_save=z_infer_sample.cpu().data.numpy()
            z_train_=z_train.cpu().data.numpy()
            y_save=y_hat.cpu().data.numpy()
            # np.savetxt("60_ae_z" +city+ ".txt", z_save)
            np.savetxt("./new/"+"60_ae_z_train" + city + ".txt", z_train_)

        #
        #
        if epoch > n_epoch-2:
            np.savetxt("loss_ae_dev_60.txt", np.array(loss_dev))
            np.savetxt("loss_ae_test_mse_60.txt", np.array(loss_test_mse))