import matplotlib.pyplot as plt
import numpy as np
import ray
import sys
sys.path.append('../gan-geosteering')
from log_gan import Gan

ray.init()

def main():
    np.random.seed(100)
    m_true = np.random.randn(60)

    keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vec_size' : 60}
    worker = Gan.remote(keys=keys)
    task = worker.call_sim.remote(input=m_true, output_field=True)
    img_true, val_true = ray.get(task)

    #states = [el['m'] for el in np.load('temp_state_iter.npz',allow_pickle=True)['arr_0'] if el is not None]
    #states = [el['m'] for el in np.load('temp_state_assim.npz',allow_pickle=True)['arr_0'] if el is not None]
    states = [el['m'] for el in np.load('temp_state_mda.npz', allow_pickle=True)['arr_0'] if el is not None]
    states.append(np.load('final.npz',allow_pickle=True)['m'])

    plt_field([img_true], 'true')

    plt_std(worker, states)
    plt_mean_diff(worker, states, img_true)
    plt_std_diff(worker, states, img_true)
    plt_data(worker,states, val_true)
    plt_mean(worker, states)


def plt_data(w, states, true_data):
    path = 'results/'
    prior = states[0]
    prior_fcs = []
    for i in range(prior.shape[1]):
        task = w.call_sim.remote(input=prior[:,i])
        val = ray.get(task)
        prior_fcs.append(val['res'])

    post_fcs = []
    post = states[-1]
    for i in range(post.shape[1]):
        task = w.call_sim.remote(input=post[:,i])
        val = ray.get(task)
        post_fcs.append(val['res'])

    for pos in range(true_data['res'].shape[0]):
        plt.figure();
        plt.plot(np.array(prior_fcs)[:,pos,:].T/true_data['res'][pos, :][:,np.newaxis],'k');
        plt.plot(np.array(post_fcs)[:, pos, :].T/true_data['res'][pos, :][:,np.newaxis],'--b');
        plt.plot(true_data['res'][pos, :]/true_data['res'][pos, :], 'r');
        plt.title(f'position {pos}');plt.savefig(path + f'data_pos_{pos}.pdf',bbox_inches='tight');plt.close()


def plt_mean_diff(w,st,img_true):
    tot_field = []
    for el in st:
        tmp_img = []
        for i in range(el.shape[1]):
            mean_f = el[:, i]
            task = w.call_sim.remote(input=mean_f, output_field=True)
            img, _ = ray.get(task)
            tmp_img.append(abs(img-img_true))
        tot_field.append(np.mean(np.array(tmp_img), axis=0))

    plt_field(tot_field,'mean_diff')

def plt_std_diff(w,st,img_true):
    tot_field = []
    for el in st:
        tmp_img = []
        for i in range(el.shape[1]):
            mean_f = el[:, i]
            task = w.call_sim.remote(input=mean_f, output_field=True)
            img, _ = ray.get(task)
            tmp_img.append(abs(img-img_true))
        tot_field.append(np.std(np.array(tmp_img), axis=0,ddof=1))

    plt_field(tot_field,'std_diff')

def plt_mean(w,st):
    tot_field = []
    for el in st:
        tmp_img = []
        for i in range(el.shape[1]):
            mean_f = el[:, i]
            task = w.call_sim.remote(input=mean_f, output_field=True)
            img, _ = ray.get(task)
            tmp_img.append(img)
        tot_field.append(np.mean(np.array(tmp_img), axis=0))

    plt_field(tot_field,'mean')


def plt_std(w, st):
    tot_field = []
    for el in st:
        tmp_img = []
        for i in range(el.shape[1]):
            mean_f = el[:,i]
            task = w.call_sim.remote(input=mean_f, output_field=True)
            img, _ = ray.get(task)
            tmp_img.append(img)
        tot_field.append(np.std(np.array(tmp_img),axis=0,ddof=1))

    plt_field(tot_field,'std')

def plt_field(val,ftype):
    path= 'results/'
    num_rows = int(np.ceil(len(val)/3))
    num_colums = int(np.ceil(len(val)/num_rows))

    plt.figure();
    for i,el in enumerate(val):
        plt.subplot(num_rows,num_colums,i+1);
        plt.imshow(el[0,:,:],cmap='jet',interpolation='none');plt.colorbar();plt.title(f'Iteration {i}');
    plt.savefig(path+ftype+'_value_chl1.pdf');plt.close()
    plt.figure();
    for i,el in enumerate(val):
        plt.subplot(num_rows, num_colums, i+1);
        plt.imshow(el[1, :, :], cmap='jet',interpolation='none');plt.colorbar();
        plt.title(f'Iteration {i}');
    plt.savefig(path + ftype+'_value_chl2.pdf');
    plt.close()






main()