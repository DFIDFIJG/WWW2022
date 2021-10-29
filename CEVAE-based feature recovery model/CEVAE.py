import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions import  normal

import setproctitle
from torch import optim
np.random.seed(0)
import seaborn as sns


n_epoch=600
torch.manual_seed(0)
np.random.seed(0)
dim_z=30
que=40
train=True
load=False
p="region_features_NYC.csv"
with open(p,encoding = 'utf-8') as f:
    inputdata_tr = np.loadtxt(f,delimiter = ",", skiprows = 1)
    inputdata_tr=inputdata_tr[:,1:]
inputdata_1=inputdata_tr
if train:
    devdata = inputdata_tr[-650:]
    batch_size_dev = 650
    batch_size = len(inputdata_tr)-batch_size_dev
    inputdata_train = inputdata_tr[:batch_size]
else:
    inputdata = np.load("ft.npy")
    batch_size = 216

# inputdata=np.load("train_65_new.npy")
# batch_size = 1952
device = 'cuda' if torch.cuda.is_available() else 'cpu'
city='Chi'
p="region_features_"+city+".csv"
with open(p,encoding = 'utf-8') as f:
    inputdata_test = np.loadtxt(f,delimiter = ",", skiprows = 1)
    inputdata_test=inputdata_test[:,1:]
all=np.concatenate((inputdata_tr,inputdata_test),axis=0)
batch_size_all=len(inputdata_tr)+len(inputdata_test)
batch_size_test=len(inputdata_test)
max_ = np.max(all, axis = 0)
min_ = np.min(all, axis = 0)
max_min=max_ - min_
max_tensor=torch.FloatTensor(max_).to(device)
min_tensor=torch.FloatTensor(min_).to(device)
max_min_tensor=torch.FloatTensor(max_min).to(device)
inputdata_train = (inputdata_train - min_) * 2 / (max_ - min_) -1
inputdata_test = (inputdata_test - min_) * 2 / (max_ - min_) -1
devdata= (devdata - min_) * 2 / (max_ - min_) -1
data_quan=np.concatenate((inputdata_train,devdata,inputdata_test),axis=0)


def normalize(A , symmetric=True):
	# A = A+I
	A = A + torch.eye(A.size(0)).to(device)

	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)

#   H(l+1)=A*H(l)*W     A:N*N  H:N*L W:L*K   H:batch_size*N*L
class GCN(nn.Module):
	'''
	Z = AXW
	'''
	def __init__(self , A, dim_in , dim_out):
		super(GCN,self).__init__()
		self.A = A
		self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
		self.fc2 = nn.Linear(dim_in,dim_in,bias=False)
		self.fc3 = nn.Linear(dim_in,dim_out,bias=False)

	def forward(self,X):

		X = F.relu(self.fc1(X.mm(self.A)))
		X = F.relu(self.fc2(X.mm(self.A)))
		return self.fc3(X.mm(self.A))

def get_accessible(A):
	B = A
	item = A
	for i in range(len(A)):
		item = np.matmul(item,A)
		B = B + item
	return B




data=np.loadtxt("best_graph_DAWG_NYC.txt")
data_prun=np.loadtxt("best_graph_prun_NYC.txt")
class p_z_(nn.Module):
    def __init__(self, dim_in=20, dim_out_con=6):
        super().__init__()
        self.input = nn.Linear(dim_in, 10)
        # dim_in is dim of latent space z
        self.output_con_mu = nn.Linear(10, dim_out_con)
        self.output_con_sigma = nn.Linear(10, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input,sig=False,output=False):
        z = F.elu(self.input(z_input))
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        if sig:
            sigma=torch.zeros_like(sigma)

        x_con = normal.Normal(mu, sigma)
        x_sample=x_con.rsample()
        if output==True:
            return x_con,x_sample,mu,sigma
        else:
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
print(list(data_re))
data_save=np.array(data_remove)[que:]

shuru1,shuru2=1.6,1.6
for inter in [(1,1)]:
    beta1, beta2 = shuru1,shuru2
    dict_re_dis = {}
    dict_re_func = {}
    dict_re_value = {}
    dict_re_mu = {}
    dict_re_sigma = {}

    dict_re_dis_dec = {}
    dict_re_func_dec = {}
    dict_re_value_dec = {}
    dict_re_mu_dec = {}
    dict_re_sigma_dec = {}

    data_dec = np.array(data_remove)[::-1]

    params = []
    data = data_prun
    for i3 in data_dec:
        index_count = np.where(data[:, i3] != 0)[0]
        num = len(index_count) + dim_z
        # print(num)
        dict_re_func_dec[i3] = p_z_(int(num), 1).to(device)
        dict_re_dis_dec[i3] = 0
        dict_re_value_dec[i3] = 0
        dict_re_mu_dec[i3] = 0
        dict_re_sigma_dec[i3] = 0
        params = params + list(dict_re_func_dec[i3].parameters())

    for i1 in data_re:
        num = data[:, i1].sum()
        print(num)
        dict_re_func[i1] = p_z_(int(num), 1).to(device)
        dict_re_dis[i1] = 0
        dict_re_value[i1] = 0
        dict_re_mu[i1] = 0
        dict_re_sigma[i1] = 0

        params = params + list(dict_re_func[i1].parameters())

    A_normed = normalize(torch.FloatTensor(data).to(device), True)
    gcn = GCN(A_normed, data.shape[1], data.shape[1]).to(device)
    p_o_z = p_z_(data.shape[1], dim_z).to(device)
    params = params + list(p_o_z.parameters()) + list(gcn.parameters())

    optimizer = optim.Adam(params, lr=5e-4)
    loss_max = -10000

    # data_last = torch.FloatTensor(inputdata).to(device)
    data_last_test = torch.FloatTensor(inputdata_test).to(device)

    data_quan = torch.FloatTensor(data_quan).to(device)
    data_last_dev = torch.FloatTensor(devdata).to(device)
    # beta1,beta2=inter
    loss_train = []
    loss_dev = []
    loss_test = []
    loss_test_all = []
    loss_test_mse = []
    for epoch in range(n_epoch):
    # data_last_dev = torch.FloatTensor(devdata).to(device)
        if train:
            # p(z)
            for i2 in data_re:
                data_will = []
                index_sam = np.where(data[:, i2] == 1)[0]
                common_now = [val for val in index_sam if val in data_save]
                common_will = [val for val in index_sam if val in data_re]
                data_now = data_quan[:, common_now]
                for inter in common_will:
                    data_will.append(dict_re_value[inter])
                for inter1 in data_will:
                    data_now = torch.cat((data_now, inter1), dim=1)
                dict_re_dis[i2],dict_re_value[i2],dict_re_mu[i2],dict_re_sigma[i2] = dict_re_func[i2](data_now,output=True)
            data_inz_all=torch.zeros_like(data_quan)
            data_inz_all[:, data_save]=data_quan[:, data_save]
            for i3 in data_re:
                data_inz_all[:, i3]= dict_re_value[i3].reshape(batch_size_all)
            data_train_mu=[]
            data_train_sigma = []
            for inter1 in data_re:
                data_train_mu.append(dict_re_mu[inter1].cpu().data.numpy())
                data_train_sigma.append(dict_re_sigma[inter1].cpu().data.numpy())

            data_train_mu=np.concatenate((data_train_mu),1)
            data_train_sigma = np.concatenate((data_train_sigma), 1)





            data_inz_all=gcn(data_inz_all)
            z_infer,z_infer_sample = p_o_z(data_inz_all)
            p_z_train =z_infer_sample
            z=normal.Normal(torch.zeros((batch_size_all,dim_z)).to(device), torch.ones((batch_size_all,dim_z)).to(device))
            z_sample=z.rsample().to(device)



            loss_1 = []
            # log q(Y|X)  2168*40
            for i2 in data_re:
                loss_1.append(dict_re_dis[i2].log_prob(dict_re_value[i2].reshape(-1,1)))
            loss_1=torch.cat(loss_1,-1)
            # log q(Z|X,Y) 2168*30
            loss_5=((z_infer.log_prob(z_infer_sample)))


            # log(Y*|X)  186*40
            loss_2=[]
            for i3 in data_re:
                loss_2.append((-dict_re_dis[i3].log_prob(data_quan)[:,i3].reshape(-1,1))[:batch_size])
            loss_2 = torch.cat(loss_2, -1)

            loss_3=-z.log_prob(z_infer_sample)

            loss_4 = []
            loss_6=[]
            for i4 in data_dec:
                data_now = z_infer_sample
                index_sam = np.where(data[:, i4] == 1)[0]
                common_now = [val for val in index_sam if val in data_save]
                common_will = [val for val in index_sam if val in data_re]

                for i5 in index_sam:
                    if i5 in common_now:
                        data_now = torch.cat((data_now, data_quan[:, i5].reshape(-1, 1)), dim=1)
                    else:
                        data_now = torch.cat((data_now, dict_re_value_dec[i5].reshape(-1, 1)), dim=1)
                dict_re_dis_dec[i4], dict_re_value_dec[i4] = dict_re_func_dec[i4](data_now)
                if i4 in data_save:
                    loss_4.append(-dict_re_dis_dec[i4].log_prob(data_quan[:, i4].reshape(-1, 1)))
                else:
                    loss_4.append(-dict_re_dis_dec[i4].log_prob(dict_re_value[i4].reshape(-1, 1)))
                    loss_6.append(-dict_re_dis_dec[i4].log_prob(data_quan[:, i4].reshape(-1, 1))[:batch_size])
            loss_4 = torch.cat(loss_4, -1)
            loss_6 = torch.cat(loss_6, -1)
            loss_mean=torch.mean(torch.sum(loss_1,1),0)+beta1*torch.mean(torch.sum(loss_2,1),0)+torch.mean(torch.sum(loss_3,1),0)+\
                      torch.mean(torch.sum(loss_4,1),0)+torch.mean(torch.sum(loss_5,1),0)+beta1*torch.mean(torch.sum(loss_6,1),0)

            objective = loss_mean
            optimizer.zero_grad()
            # Calculate gradients
            objective.backward()
            # Update step
            optimizer.step()


            # dev_loss

            for i2 in data_re:
                data_will = []
                index_sam = np.where(data[:, i2] == 1)[0]
                common_now = [val for val in index_sam if val in data_save]
                common_will = [val for val in index_sam if val in data_re]
                data_now = data_last_dev[:, common_now]
                for inter in common_will:
                    data_will.append(dict_re_value[inter])
                for inter1 in data_will:
                    data_now = torch.cat((data_now, inter1), dim=1)
                dict_re_dis[i2],dict_re_value[i2] = dict_re_func[i2](data_now)
            data_inz_all = torch.zeros_like(data_last_dev)
            data_inz_all[:, data_save] = data_last_dev[:, data_save]
            for i3 in data_re:
                data_inz_all[:, i3] = dict_re_value[i3].reshape(batch_size_dev)
            A_normed = normalize(torch.FloatTensor(data).to(device), True)
            data_inz_all = gcn(data_inz_all)

            z_infer, z_infer_sample = p_o_z(data_inz_all)
            p_z_test = z_infer_sample
            z = normal.Normal(torch.zeros((batch_size_dev, dim_z)).to(device),
                              torch.ones((batch_size_dev, dim_z)).to(device))
            z_sample = z.rsample().to(device)

            # RECONSTRUCTION LOSS

            loss_RECONSTRUCTION_dev = []

            for i4 in data_re:
                loss_RECONSTRUCTION_dev.append(
                dict_re_dis[i4].log_prob(data_last_dev[:, i4].reshape(-1, 1)).sum(1))



            # test_loss
            for i2 in data_re:
                data_will = []
                index_sam = np.where(data[:, i2] == 1)[0]
                common_now = [val for val in index_sam if val in data_save]
                common_will = [val for val in index_sam if val in data_re]
                data_now = data_last_test[:, common_now]
                for inter in common_will:
                    data_will.append(dict_re_value[inter])
                for inter1 in data_will:
                    data_now = torch.cat((data_now, inter1), dim=1)
                dict_re_dis[i2],dict_re_value[i2],dict_re_mu[i2],dict_re_sigma[i2] = dict_re_func[i2](data_now,output=True)
            data_inz_all=torch.zeros_like(data_last_test)
            data_inz_all[:, data_save]=data_last_test[:, data_save]
            for i3 in data_re:
                data_inz_all[:, i3]= dict_re_value[i3].reshape(batch_size_test)
            A_normed = normalize(torch.FloatTensor(data).to(device))
            data_inz_all=gcn(data_inz_all)

            z_infer,z_infer_sample = p_o_z(data_inz_all)
            p_z_test =z_infer_sample
            z=normal.Normal(torch.zeros((batch_size_test,dim_z)).to(device), torch.ones((batch_size_test,dim_z)).to(device))
            z_sample=z.rsample().to(device)
            print("loss_train:", loss_mean)
            print("loss_RECONSTRUCTION_dev:", torch.mean(np.sum(loss_RECONSTRUCTION_dev, axis=0)))
            # RECONSTRUCTION LOSS

            loss_RECONSTRUCTION_test = []
            data_test=[]
            data_test_mu=[]
            data_test_sigma=[]
            loss_RECONSTRUCTION_test_new = []
            for i4 in data_dec:
                 data_now = z_infer_sample
                 index_sam = np.where(data[:, i4] == 1)[0]
                 for i5 in index_sam:
                     data_now = torch.cat((data_now, dict_re_value_dec[i5]), dim=1)
                 dict_re_dis_dec[i4],dict_re_value_dec[i4],dict_re_mu_dec[i4],dict_re_sigma_dec[i4] = dict_re_func_dec[i4](data_now,output=True)
                 loss_RECONSTRUCTION_test.append(dict_re_dis_dec[i4].log_prob(data_last_test[:, i4].reshape(-1, 1)))
            loss_RECONSTRUCTION_test_ch=torch.cat(loss_RECONSTRUCTION_test,1)
            loss_j=loss_RECONSTRUCTION_test_ch.cpu().data.numpy()
            loss_j=np.sum(loss_j,axis=0)

            # for ii in range(60):
            #     data_test.append(dict_re_value_dec[ii].cpu().data.numpy())
            for inter1 in data_re:
                data_test_mu.append(dict_re_mu[inter1].cpu().data.numpy())
                data_test_sigma.append(dict_re_sigma[inter1].cpu().data.numpy())
                data_test.append(dict_re_value[inter1].cpu().data.numpy())
                loss_RECONSTRUCTION_test_new.append(dict_re_dis[inter1].log_prob(data_last_test[:, inter1].reshape(-1, 1)))
            loss_RECONSTRUCTION_test_new = torch.cat(loss_RECONSTRUCTION_test_new, 1)
            loss_RECONSTRUCTION_test_new = torch.mean(torch.sum(loss_RECONSTRUCTION_test_new, 1))
            print("loss_RECONSTRUCTION_test:", loss_RECONSTRUCTION_test_new)
            data_test_mu=np.concatenate((data_test_mu),1)
            data_test_sigma = np.concatenate((data_test_sigma), 1)
            y_hat=np.concatenate((data_test),1)
            # np.savetxt("cvae_u.txt", data_test_mu)
            # np.savetxt("cvae_sigma.txt",data_test_sigma)
            # y_hat=np.array(data_test).reshape(1952,60)
            criterion = nn.MSELoss()
            y_hat=torch.FloatTensor(data_test_mu).to(device)
            # y_hat=y_hat*max_min_tensor+min_tensor
            # data_use_test=data_last_test*max_min_tensor+min_tensor
            # loss_RECONSTRUCTION_test = (-torch.abs(dict_re_dis_dec.log_prob(data_last_test)).sum(1))
            loss_RECONSTRUCTION_test_mse = criterion(y_hat, data_last_test[:,list(data_re)])
            print("loss_RECONSTRUCTION_test_mse:", loss_RECONSTRUCTION_test_mse)
            # print("loss_REGULARIZATION:", torch.mean(loss_REGULARIZATION))
            # print("loss_AUXILIARY:", torch.mean(np.sum(loss_AUXILIARY, axis=0)))



            # loss_cal=torch.mean(np.sum(loss_RECONSTRUCTION_dev, axis=0))

            loss_train.append(float(-loss_mean.cpu().data.numpy()))
            loss_dev.append(float(torch.mean(np.sum(loss_RECONSTRUCTION_dev, axis=0)).cpu().data.numpy()))
            loss_test.append(float(loss_RECONSTRUCTION_test_new.cpu().data.numpy()))
            loss_test_mse.append(float(loss_RECONSTRUCTION_test_mse.cpu().data.numpy()))
            if  float(torch.mean(np.sum(loss_RECONSTRUCTION_dev, axis=0)).cpu().data.numpy())>loss_max:
                loss_max=float(torch.mean(np.sum(loss_RECONSTRUCTION_dev, axis=0)).cpu().data.numpy())
                best_loss_RECONSTRUCTION_test=loss_RECONSTRUCTION_test_new
                print("best:",best_loss_RECONSTRUCTION_test)
                p_z_train = p_z_train.cpu().data.numpy()
                p_z_test = p_z_test.cpu().data.numpy()
                np.savetxt("./new/"+"60_cvae_u"+ str(que)+city+"dim_z="+str(dim_z)+"beta1="+str(beta1)+"beta2="+str(beta2)+ ".txt", data_test_mu)
                np.savetxt("./new/"+"60_cvae_sigma"+ str(que)+city+"dim_z="+str(dim_z)+"beta1="+str(beta1)+"beta2="+str(beta2)+ ".txt",data_test_sigma)
                np.savetxt("./new/"+"60_cvae_u_train" + str(que) + city +"dim_z="+str(dim_z)+"beta1="+str(beta1)+"beta2="+str(beta2)+ ".txt", data_train_mu)
                np.savetxt("./new/"+"60_cvae_sigma_train" + str(que) + city +"dim_z="+str(dim_z)+"beta1="+str(beta1)+"beta2="+str(beta2)+ ".txt", data_train_sigma)
                # np.savetxt("60_dimz=" + str(dim_z) + "_" + "train_cvae_graph" + str(que) + ".txt", p_z_train)
                # np.savetxt("60_dimz=" + str(dim_z) + "_" + "test_cvae_graph" + str(que) + ".txt", p_z_test)
                # for i3 in data_dec:
                #     torch.save(dict_re_func_dec[i3], './net_dec_65/'+str(dim_z)+"_"+str(que)+str(i3)+".pkl")
                # for i1 in data_re:
                #     torch.save(dict_re_func[i1], './net_enc_65/' +str(dim_z)+"_"+str(que)+ str(i1) + ".pkl")
                # torch.save(p_o_z, './net_enc_65/'+str(dim_z) +"_"+str(que)+ "_z" + ".pkl")
            if epoch>(n_epoch-2):

                np.savetxt("loss_cvae_train_60.txt",np.array(loss_train))
                np.savetxt("loss_cvae_dev_60.txt", np.array(loss_dev))
                np.savetxt("loss_cvae_test_60.txt", np.array(loss_test))
                np.savetxt("loss_cvae_test_mse_60.txt", np.array(loss_test_mse))
                index=np.where(np.array(loss_dev)==np.max(loss_dev))
                data_append_loglikly=np.array(loss_test)[index]
                data_append_mse = np.array(loss_test_mse)[index]
                with open('result_ablation_loglikely.txt', 'a+') as f:
                    np.savetxt(f, data_append_loglikly)
                with open('result_ablation_mse.txt', 'a+') as f:
                    np.savetxt(f, data_append_mse)


        else:

                # test_loss
                batch_size_test = 1952
                for i2 in data_re:
                    data_will = []
                    index_sam = np.where(data[:, i2] == 1)[0]
                    common_now = [val for val in index_sam if val in data_save]
                    common_will = [val for val in index_sam if val in data_re]
                    data_now = data_last_test[:, common_now]
                    for inter in common_will:
                        data_will.append(dict_re_value[inter])
                    for inter1 in data_will:
                        data_now = torch.cat((data_now, inter1), dim=1)
                    dict_re_value[i2] = dict_re_func[i2](data_now)[1]
                data_inz_all = torch.zeros_like(data_last_test)
                data_inz_all[:, data_save] = data_last_test[:, data_save]
                for i3 in data_re:
                    data_inz_all[:, i3] = dict_re_value[i3].reshape(batch_size_test)
                A_normed = normalize(torch.FloatTensor(data).to(device))
                data_inz_all = gcn(data_inz_all)

                z_infer, z_infer_sample = p_o_z(data_inz_all)
                p_z_test = z_infer_sample
                z = normal.Normal(torch.zeros((batch_size_test, dim_z)).to(device),
                                  torch.ones((batch_size_test, dim_z)).to(device))
                z_sample = z.rsample().to(device)

                # RECONSTRUCTION LOSS

                loss_RECONSTRUCTION_test = []
                data_test = []
                for i4 in data_dec:
                    data_now = z_infer_sample
                    index_sam = np.where(data[:, i4] == 1)[0]
                    for i5 in index_sam:
                        data_now = torch.cat((data_now, dict_re_value_dec[i5]), dim=1)
                    dict_re_dis_dec[i4], dict_re_value_dec[i4] = dict_re_func_dec[i4](data_now)
                    loss_RECONSTRUCTION_test.append(
                        -dict_re_dis_dec[i4].log_prob(data_last_test[:, i4].reshape(-1, 1)).sum(1))
                loss_RECONSTRUCTION_test=torch.mean(np.sum(loss_RECONSTRUCTION_test,axis=0))
                print("loss_RECONSTRUCTION_test:", loss_RECONSTRUCTION_test)
                loss_test_all.append(float(torch.mean(loss_RECONSTRUCTION_test).cpu().data.numpy()))
             