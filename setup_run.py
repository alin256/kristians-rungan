import numpy as np
import ray
import sys

sys.path.append('../gan-geosteering')
from log_gan import Gan
import csv
ray.init()


def main():
    np.random.seed(100)
    m_true = np.random.randn(60) * 0.01
    
    keys = {'bit_pos' : [0, 4, 8, 12, 16, 20, 24, 28, 32],
            'vec_size' : 60}
    worker = Gan.remote(keys=keys)
    task = worker.call_sim.remote(input=m_true, output_field=True)
    img,val = ray.get(task)

    with open('true_data.csv','w') as f:
        writer = csv.writer(f)
        for el in range(len(keys['bit_pos'])):
            writer.writerow([str(elem) for elem in val['res'][el,:]])

    with open('data_index.csv','w') as f:
        writer = csv.writer(f)
        for el in keys['bit_pos']:
            writer.writerow([str(el)])

main()