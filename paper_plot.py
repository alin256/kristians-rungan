import matplotlib.pyplot as plt
# plt.style.use('plots.mplstyle')
import numpy as np
import ray
import sys
sys.path.append('../gan-geosteering')
from log_gan import Gan
sys.path.append('../deepEMdeepML2/deep-borehole-inverse-problem/KERNEL')
sys.path.append('../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append('../gan-geosteering')
from resitivity import get_resistivity_default

convert = get_resistivity_default

ray.init()


keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vec_size' : 60}
worker = Gan.remote(keys=keys)

def precalc():
    m_final = np.load('final.npz')['m']
    tot_image = []
    tot_res = []

    for i in range(500):
        tmp_res = []
        m = m_final[:,i]
        task = worker.call_sim.remote(input=m, output_field=True)
        img, val = ray.get(task)
        for pos in range(64):
            pixel_input = np.array(img[:,:,pos])
            tmp_res.append([convert(pix) for pix in pixel_input.T])
        tot_image.append(img)
        tot_res.append(tmp_res)

    np.random.seed(100)
    m_prior = np.dot(np.linalg.cholesky(1e-6*np.eye(60)),np.random.randn(60,3))
    tot_pr_image = []
    tot_pr_res = []

    for i in range(3):
        tmp_res = []
        m = m_prior[:,i]
        task = worker.call_sim.remote(input=m, output_field=True)
        img, val = ray.get(task)
        for pos in range(64):
            pixel_input = np.array(img[:,:,pos])
            tmp_res.append([convert(pix) for pix in pixel_input.T])
        tot_pr_image.append(img)
        tot_pr_res.append(tmp_res)

    np.random.seed(0)
    m_true = np.dot(np.linalg.cholesky(1e-6*np.eye(60)),np.random.randn(60,1))
    tot_true_image = []
    tot_true_res = []
    tmp_res = []
    task = worker.call_sim.remote(input=m_true, output_field=True)
    img, val = ray.get(task)
    for pos in range(64):
        pixel_input = np.array(img[:,:,pos])
        tmp_res.append([convert(pix) for pix in pixel_input.T])
    tot_true_image.append(img)
    tot_true_res.append(tmp_res)


    # tot_mcmc_img = np.load('mcmc_samples.npz')['tot_mcmc_img']
    # tot_mcmc_res = []
    # for i in range(1600):
    #     tmp_res = []
    #     img = tot_mcmc_img[i,:,:,:]
    #     for pos in range(64):
    #         pixel_input = np.array(img[:,:,pos])
    #         tmp_res.append([convert(pix) for pix in pixel_input.T])
    #     tot_mcmc_res.append(tmp_res)
    #
    # np.savez('foobar.npz',**{'res':tot_res, 'img':tot_image,'pr_res':tot_pr_res,'pr_img':tot_pr_image,'mcmc_res':tot_mcmc_res,'mcmc_img':tot_mcmc_img,'true_img':tot_true_image,'true_res':tot_true_res})

def plot_images():
    path = 'upd_plots/'
    f = np.load('foobar.npz')
    content = f.files
    tot_mcmc_indx = [0,220,499]
    print(tot_mcmc_indx)
    for el in content:
        tot_val = f[el]
        if 'img' in el:
            for i in range(3):
                plt.figure();plt.imshow(tot_val[:,i,:,:].mean(axis=0),interpolation='none');plt.colorbar();
                plt.plot([0,1,2,3,4,5,6,7,8],[32,32,32,32,32,32,32,32,32],'*r');plt.savefig(path+f'{el}_mean_chl_{i}.png',bbox_inches='tight');plt.close()
                plt.figure();plt.imshow(tot_val[:,i,:,:].std(ddof=1,axis=0),interpolation='none');plt.colorbar();
                plt.plot([0,1,2,3,4,5,6,7,8],[32,32,32,32,32,32,32,32,32],'*r');plt.savefig(path+f'{el}_std_chl_{i}.png',bbox_inches='tight');plt.close()
                if 'true' in el:
                    num_samp = 1
                else:
                    num_samp = 3
                for j in range(num_samp):
                    if tot_val.shape[0] > num_samp:
                        indx = tot_mcmc_indx[j]
                    else:
                        indx = j
                    plt.figure();plt.imshow(tot_val[indx,i,:,:],interpolation='none');plt.colorbar();
                    plt.plot([0,1,2,3,4,5,6,7,8],[32,32,32,32,32,32,32,32,32],'*r');plt.savefig(path+f'{el}_real_{j}_chl_{i}.png',bbox_inches='tight');plt.close()

        else:
            plt.figure();plt.imshow(tot_val.mean(axis=0).T,interpolation='none');plt.colorbar();
            plt.plot([0,1,2,3,4,5,6,7,8],[32,32,32,32,32,32,32,32,32],'*r');plt.savefig(path+f'{el}_mean.png',bbox_inches='tight');plt.close()
            plt.figure();plt.imshow(tot_val.std(ddof=1,axis=0).T,interpolation='none');plt.colorbar();
            plt.plot([0,1,2,3,4,5,6,7,8],[32,32,32,32,32,32,32,32,32],'*r');plt.savefig(path+f'{el}_std.png',bbox_inches='tight');plt.close()
            if 'true' in el:
                num_samp = 1
            else:
                num_samp = 3
            for j in range(num_samp):
                if tot_val.shape[0] > num_samp:
                    indx = tot_mcmc_indx[j]
                else:
                    indx = j
                print(f'ploting {el} realization {indx}')
                plt.figure();plt.imshow(tot_val[indx,:,:].T,interpolation='none');plt.colorbar();
                plt.plot([0,1,2,3,4,5,6,7,8],[32,32,32,32,32,32,32,32,32],'*r');plt.savefig(path+f'{el}_real_{j}.png',bbox_inches='tight');plt.close()

precalc()
plot_images()
