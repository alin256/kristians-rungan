import numpy as np
from scipy.linalg import block_diag

obs = np.load('obs_var.npz',allow_pickle=True)['obs']
data = np.array([[obs[i][f'res{el}'] for el in range(1, 14)] for i in range(9)]).flatten()

rel_var = 0.01

corrRange = 10
nD=13 # only correlation along point data
sub_covD = np.zeros((9,nD,nD))
for k in range(9):
    sub_data = np.array([obs[k][f'res{el}'] for el in range(1, 14)]).flatten()
    for i in range(nD):
        for j in range(nD):
            sub_covD[k,i,j] = (sub_data[i]*rel_var)*(sub_data[j]*rel_var) * np.exp(-3.*(np.abs(i - j)/corrRange)**1.9)
Cd = block_diag(sub_covD[0,:,:],sub_covD[1,:,:], sub_covD[2,:,:],sub_covD[3,:,:],sub_covD[4,:,:],
                             sub_covD[5,:,:],sub_covD[6,:,:], sub_covD[7,:,:],sub_covD[8,:,:])

np.savez('cd.npz',Cd)
