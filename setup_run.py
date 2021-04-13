import numpy as np
import ray
import sys
import matplotlib.pyplot as plt

sys.path.append('../gan-geosteering')
from log_gan import Gan
import csv

ray.init()


def main_script():
    # np.random.seed(100)
    np.random.seed(0)
    numpy_input = np.load('../gan-geosteering/saves/chosen_realization_C1.npz')
    numpy_single = numpy_input['arr_0']
    m_true = numpy_single.copy()

    keys = {'bit_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vec_size': 60}
    worker = Gan.remote(keys=keys)
    task = worker.call_sim.remote(input=m_true, output_field=True)
    img, val = ray.get(task)
    plt.imshow(img[0, :, :], interpolation='none')
    plt.colorbar()
    plt.savefig('True_GAN_field.png')
    plt.close()

    np.savez('true_img.npz', img)

    with open('true_data.csv', 'w') as f:
        writer = csv.writer(f)
        for el in range(len(keys['bit_pos'])):
            writer.writerow([str(elem) for elem in val['res'][el, :]])

    with open('data_index.csv', 'w') as f:
        writer = csv.writer(f)
        for el in keys['bit_pos']:
            writer.writerow([str(el)])

    mean_f = m_true * 0.6
    np.savez('mean_field.npz', mean_f)



if __name__ == '__main__':
    main_script()
