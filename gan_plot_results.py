import numpy as np
import sys
import ray
import matplotlib.pyplot as plt
import setup_run
sys.path.append('../gan-geosteering')
from log_gan import Gan


def get_true():
    numpy_input = np.load('../gan-geosteering/saves/chosen_realization_C1.npz')
    numpy_single = numpy_input['arr_0']
    m_true = numpy_single.copy()
    return m_true


def get_prior():
    setup_run.main_script()
    prior_mean = np.load('mean_field.npz', allow_pickle=True)['arr_0']
    np.random.seed(0)
    # m_prior = np.dot(prior_mean[:, np.newaxis], np.ones((1, 500))) + np.dot(np.linalg.cholesky(0.6 * np.eye(60)),
    #                                                                         np.random.randn(60, 500))
    # m_prior = np.dot(np.linalg.cholesky(0.6 * np.eye(60)), np.random.randn(60, 500))
    m_prior = np.dot(prior_mean[:, np.newaxis], np.ones((1, 500))) \
            + np.dot(np.linalg.cholesky(0.9 * np.eye(60)), np.random.randn(60, 500))
    return m_prior


def get_posterior():
    post_save = np.load('final.npz', allow_pickle=True)['m']
    return post_save


def plot_sand_probability(ensemble, label=''):
    posterior_earh_models = (ensemble + 1.)/2.
    mean_model = np.mean(posterior_earh_models, 0)
    plt.figure()
    plt.imshow(1.-mean_model[0, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    plt.title('Probability of sand ({})'.format(label))
    plt.colorbar()

    plt.figure()
    plt.imshow(mean_model[1, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    plt.title('Probability of good sand ({})'.format(label))
    plt.colorbar()

    # plt.figure()
    # plt.imshow(mean_model[2, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    # plt.title('Probability of 2')
    # plt.colorbar()


if __name__ == '__main__':
    ray.init()
    true = get_true()
    prior = get_prior()
    posterior = get_posterior()

    keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vec_size': 60}


    worker = Gan.remote(keys=keys)
    task_prior = worker.generate_earth_model.remote(input=prior)
    task_posterior = worker.generate_earth_model.remote(input=posterior)

    prior_earth_model = ray.get(task_prior)
    posterior_earth_model = ray.get(task_posterior)

    plot_sand_probability(prior_earth_model, label='less informed prior')
    # plot_sand_probability(posterior_earth_model, label='posterior')

    plt.show()

