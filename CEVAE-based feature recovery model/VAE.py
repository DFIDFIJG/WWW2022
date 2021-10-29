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
data_quan_all=np.concatenate((inputdata_train,devdata,inputdata_test),axis=0)
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
        self.output_con_mu = nn.Linear(10, dim_out_con)
        self.output_con_sigma = nn.Linear(10, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input,sig1=False,sig=False):
        z = F.elu(self.input(z_input))
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        if sig1:
            sigma=torch.zeros_like(sigma)
        # mu=mu+ba
        if sig:
            return mu,sigma
        else:
            x_con = normal.Normal(mu, sigma)
            x_sample = x_con.rsample()



            return x_con,x_sample

global data1
data1=list(data)

data_sum0=np.sum(data,axis=0)
node_father=np.where(data_sum0==0)



data_sum1=np.sum(data,axis=1)
node_child=np.where(data_sum1==0)




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

dict_re_func_dec=p_z_(int(dim_z), int(len(data))).to(device)

params=params+list(dict_re_func_dec.parameters())




loss_test_all=[]
p_o_z = p_z_(int(len(data_save)), dim_z).to(device)
params=params+list(p_o_z.parameters())

optimizer = optim.Adam(params, lr=1e-3)
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
        z=normal.Normal(torch.zeros((batch_size_all,dim_z)).to(device), torch.ones((batch_size_all,dim_z)).to(device))
        z_sample=z.rsample().to(device)
        z_infer,z_infer_sample = p_o_z(data_quan[:,data_save])
        p_z_train=z_infer_sample
        z_infer_1, z_infer_sample_1 = p_o_z(data_quan_all[:, data_save])
        p_z_train_1 = z_infer_sample_1
        #

        # RECONSTRUCTION LOSS

        data_now = z_infer_sample
        dict_re_dis_dec = dict_re_func_dec(data_now)[0]
        train_mu, train_sigma = dict_re_func_dec(z_infer_sample_1, sig=True)
        train_mu=train_mu[:2167]
        train_sigma=train_sigma[:2167]
        loss_RECONSTRUCTION=(dict_re_dis_dec.log_prob(data_quan))[:batch_size].sum(1)


        train_mu = train_mu.cpu().data.numpy()
        train_sigma = train_sigma.cpu().data.numpy()
        # loss_RECONSTRUCTION = []


        # REGULARIZATION LOSS
        # p(z) - q(z|o)
        # loss_REGULARIZATION=[]
        loss_REGULARIZATION=(z.log_prob(z_infer_sample)-z_infer.log_prob(z_infer_sample)).sum(1)

        # loss_REGULARIZATION.append(((-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)))

        # loss_mean = torch.mean(np.sum(loss_RECONSTRUCTION,axis=0)  + np.sum(loss_AUXILIARY,axis=0))+torch.mean(loss_REGULARIZATION)
        loss_mean = -torch.mean(loss_RECONSTRUCTION) - torch.mean(loss_REGULARIZATION)

        print("loss_RECONSTRUCTION:", torch.mean(loss_RECONSTRUCTION))
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
        z_infer, z_infer_sample = p_o_z(data_inz_all)
        p_z_test = z_infer_sample
        data_now = z_infer_sample
        dict_re_dis_dec = dict_re_func_dec(data_now)[0]
        loss_RECONSTRUCTION_dev = (-dict_re_dis_dec.log_prob(data_last_dev).sum(1))

        loss_cal=-torch.mean(loss_RECONSTRUCTION_dev)
        print("loss_dev:",loss_cal)


        # test_loss
        data_inz_all = data_last_test[:, data_save]
        z_infer_sample = p_o_z(data_inz_all)[1]
        # p_z_test = z_infer_sample
        data_now = z_infer_sample
        dict_re_dis_dec = dict_re_func_dec(data_now)[0]
        test_mu,test_sigma = dict_re_func_dec(data_now,sig=True)
        y_hat=test_mu
        test_mu=test_mu.cpu().data.numpy()
        test_sigma = test_sigma.cpu().data.numpy()
        # y_hat=dict_re_func_dec(data_now)[1]
        criterion = nn.MSELoss()
        loss_RECONSTRUCTION_test = (dict_re_dis_dec.log_prob(data_last_test)[:,list(data_re)].sum(1))
        loss_RECONSTRUCTION_test_mse=criterion(y_hat[:,list(data_re)],data_last_test[:,list(data_re)])
        # loss_train.append(float(torch.mean(loss_RECONSTRUCTION).cpu().data.numpy()))
        loss_dev.append(float(torch.mean(loss_RECONSTRUCTION_dev).cpu().data.numpy()))
        # # loss_test.append(float(torch.mean(loss_RECONSTRUCTION_test).cpu().data.numpy()))
        loss_test.append(float(torch.mean(loss_RECONSTRUCTION_test).cpu().data.numpy()))
        loss_test_mse.append(float(loss_RECONSTRUCTION_test_mse.cpu().data.numpy()))
        print("loss_RECONSTRUCTION_test:", torch.mean(loss_RECONSTRUCTION_test))
        print("loss_RECONSTRUCTION_test_mse:", loss_RECONSTRUCTION_test_mse)
        if loss_cal>loss_max:
            loss_max=loss_cal
            best_loss_RECONSTRUCTION_test = torch.mean(loss_RECONSTRUCTION_test)
            # best_loss_RECONSTRUCTION_test=0
            # np.savetxt("60_vae_u.txt", test_mu)
            # np.savetxt("60_vae_sigma.txt",test_sigma)
            np.savetxt("./new/"+"60_vae_u_train"+city+".txt", train_mu)
            np.savetxt("./new/"+"60_vae_sigma_train"+city+".txt", train_sigma)
            print("best:", best_loss_RECONSTRUCTION_test)




        if epoch > n_epoch-2:
            np.savetxt("loss_vae_train_60.txt", np.array(loss_train))
            np.savetxt("loss_vae_dev_60.txt", np.array(loss_dev))
            np.savetxt("loss_vae_test_60.txt", np.array(loss_test))
            np.savetxt("loss_vae_test_mse_60.txt", np.array(loss_test_mse))


    else:
        # for i1 in data_re:
        #     dict_re_func[i1]=torch.load('./net_enc_65/' +str(dim_z)+"_"+str(que)+ str(i1) + ".pkl")
        p_o_z=torch.load('./net_enc_65/'+str(dim_z) +"_"+"only"+str(que)+ "_z" + ".pkl")
        dict_re_func_dec=torch.load('./net_dec_65/' + str(dim_z) + "_" + "only" + str(que) + str(0) + ".pkl")
        batch_size_test = 1952
        data_inz_all = data_last_test[:, data_save]
        z_infer, z_infer_sample = p_o_z(data_inz_all,True)
        p_z_test = z_infer_sample
        data_now = z_infer_sample
        dict_re_dis_dec = dict_re_func_dec(data_now,True)[0]
        y_hat = dict_re_func_dec(data_now,True)[1]
        criterion = nn.MSELoss()
        loss_RECONSTRUCTION_test = (-torch.abs(dict_re_dis_dec.log_prob(data_last_test)).sum(1))
        loss_test_all.append( float(torch.mean(loss_RECONSTRUCTION_test).cpu().data.numpy()))
        loss_RECONSTRUCTION_test_mse = criterion(y_hat, data_last_test)
        if epoch > 98:
            print(np.log(-np.mean(loss_test_all)))
        print("loss_RECONSTRUCTION_test:", torch.mean(loss_RECONSTRUCTION_test))