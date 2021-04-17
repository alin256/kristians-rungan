# Imports
import sys
sys.path.insert(0,'../pipt_gan') # ray implementation
import ip
import numpy as np

def main():
    # Run inversion
    #random.seed(5)
    np.random.seed(0)
    ip.run_inversion('GAN_test.txt', 'final')
    # ip.run_inversion('GAN_step2.txt', 'final2')
if __name__ == '__main__':
    main()
