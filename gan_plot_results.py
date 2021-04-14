import numpy as np
import sys
import ray
import matplotlib.pyplot as plt
sys.path.append('../gan-geosteering')
from log_gan import Gan


def get_true():
    numpy_input = np.load('../gan-geosteering/saves/chosen_realization_C1.npz')
    numpy_single = numpy_input['arr_0']
    m_true = numpy_single.copy()
    return m_true

def get_prior():
    prior_mean = np.load('mean_field.npz', allow_pickle=True)['arr_0']
    np.random.seed(0)
    m_prior = np.dot(prior_mean[:, np.newaxis], np.ones((1, 500))) \
              + np.dot(np.linalg.cholesky(0.6 * np.eye(60)), np.random.randn(60, 500))
    return m_prior


def get_posterior():
    post_save = np.load('final.npz', allow_pickle=True)['m']
    return post_save


if __name__ == '__main__':
    true = get_true()
    prior = get_prior()
    posterior = get_posterior()

    keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vec_size': 60}

    ray.init()
    worker = Gan.remote(keys=keys)
    task = worker.generate_earth_model.remote(input=prior)
    posterior_earh_models = ray.get(task)
    posterior_earh_models = (posterior_earh_models + 1.)/2.
    mean_model = np.mean(posterior_earh_models, 0)

    plt.figure()
    plt.imshow(1.-mean_model[0, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='summer')
    plt.title('Probability of sand')
    plt.colorbar()

    plt.figure()
    plt.imshow(mean_model[1, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='summer')
    plt.title('Probability of 1')
    plt.colorbar()

    plt.figure()
    plt.imshow(mean_model[2, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='summer')
    plt.title('Probability of 2')
    plt.colorbar()

    plt.show()

