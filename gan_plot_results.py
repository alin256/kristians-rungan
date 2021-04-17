import numpy as np
import sys
import ray
import matplotlib
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
            + np.dot(np.linalg.cholesky(1.0 * np.eye(60)), np.random.randn(60, 500))
    return m_prior


def get_posterior(filename='final.npz'):
    post_save = np.load(filename, allow_pickle=True)['m']
    return post_save


def save_plot(name):
    plt.savefig('eage/{}.png'.format(name), bbox_inches='tight', dpi=600)
    plt.savefig('eage/{}.pdf'.format(name), bbox_inches='tight')


def plot_resistivity(resistivity_ensemble, label='', plot_realizatoins=False):
    if len(resistivity_ensemble.shape) == 3:
        mean_model = np.mean(resistivity_ensemble, 0)
    else:
        mean_model = resistivity_ensemble
    plt.figure()
    plt.imshow(mean_model, interpolation='none', vmin=1.,vmax=150., cmap='summer')
    plt.title('Resistivity mean ({}), ohm m'.format(label))
    plt.colorbar()

    if len(resistivity_ensemble.shape) == 3:
        std_model = np.std(resistivity_ensemble, axis=0)
        plt.figure()
        plt.imshow(std_model, interpolation='none', vmin=1.,vmax=150., cmap='summer')
        plt.title('Resistivity std ({}), ohm m'.format(label))
        plt.colorbar()

    if len(resistivity_ensemble.shape) == 3 and plot_realizatoins:
        for i in range(6):
            plt.figure()
            plt.imshow(resistivity_ensemble[i, :, :], vmin=1., vmax=150., cmap='summer')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axes.set_aspect(0.2)
            save_plot('resistivity_{}'.format(i))
            # plt.title('Facies type')
            plt.close()


def plot_sand_probability(ensemble, label='', plot_realizatoins=False):
    posterior_earh_models = (ensemble + 1.)/2.
    if len(ensemble.shape) == 4:
        mean_model = np.mean(posterior_earh_models, 0)
    else:
        mean_model = np.round(ensemble)
    plt.figure()
    plt.imshow(1.-mean_model[0, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    plt.title('Probability of sand ({})'.format(label))
    plt.colorbar()

    plt.figure()
    plt.imshow(mean_model[1, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    plt.title('Probability of good sand ({})'.format(label))
    plt.colorbar()

    if len(ensemble.shape) == 4 and plot_realizatoins:
        index_image = np.argmax(ensemble, 1)
        for i in range(6):
            plt.figure()
            plt.imshow(index_image[i, :, :], cmap='Paired')
            # plt.title('Facies type')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axes.set_aspect(0.2)
            save_plot('facies_{}'.format(i))
            plt.close()

    # plt.figure()
    # plt.imshow(mean_model[2, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    # plt.title('Probability of 2')
    # plt.colorbar()


def plot_logs(ensemble, worker, indeces=[5,6,7,8,9,10,11,12]):
    for i in range(6):
        task = worker.call_sim.remote(input=ensemble[:, i])
        logs = ray.get(task)['res']
        plt.figure(figsize=(10, 2))
        plt.plot(logs[:, indeces])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        save_plot('logs_{}'.format(i))
        plt.close()

if __name__ == '__main__':
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    ray.init()
    true = get_true()
    prior = get_prior()
    # posterior = get_posterior('final_8.npz')
    posterior = get_posterior('final.npz')
    posterior2 = get_posterior('final2.npz')

    keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vec_size': 60}

    worker = Gan.remote(keys=keys)

    plot_logs(posterior, worker)

    task_true = worker.generate_earth_model.remote(input=true)
    task_prior = worker.generate_earth_model.remote(input=prior)
    task_posterior = worker.generate_earth_model.remote(input=posterior)
    task_posterior2 = worker.generate_earth_model.remote(input=posterior2)

    true_earth_model = ray.get(task_true)
    prior_earth_model = ray.get(task_prior)
    posterior_earth_model = ray.get(task_posterior)
    posterior_earth_model2 = ray.get(task_posterior2)

    task_convert = worker.convert_to_resistivity_poor.remote(prior_earth_model)
    converted_prior = ray.get(task_convert)

    task_convert = worker.convert_to_resistivity_poor.remote(posterior_earth_model)
    converted_posterior = ray.get(task_convert)

    plot_resistivity(converted_prior, label='prior')
    plot_resistivity(converted_posterior, label='posterior', plot_realizatoins=True)
    #plt.show()




    plot_sand_probability(true_earth_model, label='true model')
    plot_sand_probability(prior_earth_model, label='less informed prior')
    plot_sand_probability(posterior_earth_model, label='posterior', plot_realizatoins=True)
    # plot_sand_probability(posterior_earth_model2, label='posterior2')
    plt.show()


