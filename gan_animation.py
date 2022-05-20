import os
import matplotlib
import matplotlib.pyplot as plt

import ray

global_extent = [-90, 550, -16.25, 15.75]


def save_plot(name, folder='gif'):
    plt.savefig('{}/{}.png'.format(folder, name), bbox_inches='tight', dpi=200)
    # plt.savefig('example2-enrml/{}.pdf'.format(name), bbox_inches='tight')


def _plot_resistivity(resistivity_ensemble, label='anim', grid_x=2, grid_y=3, folder='gif'):
    norm = matplotlib.colors.Normalize(vmin=1., vmax=200.0)

    fig, axs = plt.subplots(grid_y, grid_x, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 4))
    # plt.figure(figsize=(10, 4))

    for i in range(grid_y):
        for j in range(grid_x):
            ind = i * grid_x + j
            realizatoin = resistivity_ensemble[ind]
            ax = axs[i, j]
            pcm = ax.imshow(resistivity_ensemble[ind, :, :], norm=norm, cmap='summer', extent=global_extent)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axes.set_aspect(4)
            # plt.title('Facies type')

    fig.colorbar(pcm, ax=axs[:, grid_x - 1], shrink=0.6)
    save_plot('resisitivity_anim_{}'.format(label), folder=folder)
    plt.close(fig)


def create_animation(model_vectors, gan_worker,
                     folder='gif',
                     grid_x=6,
                     grid_y=6,
                     frames=40,
                     changed_components=[0, 2, 4, 8, 10, 12]):
    try:
        os.makedirs(folder)
    except:
        print('Cannot create folder {}. Already there?'.format(folder))

    first_frame = 100
    for frame in range(frames):
        new_value = 2. * (frame + 0.0 - frames / 2) / frames
        for comp in changed_components:
            model_vectors[comp, :] = new_value
        task_prior = gan_worker.generate_earth_model.remote(input=model_vectors)
        prior_earth_model = ray.get(task_prior)
        task_convert = gan_worker.convert_to_resistivity_poor.remote(prior_earth_model)
        converted_prior = ray.get(task_convert)

        _plot_resistivity(converted_prior, label=str(first_frame + frame), folder=folder)


