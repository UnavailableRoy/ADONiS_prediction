

import pandas as pd
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt
import sys

from train_wm import wang_mendel
from Testing import testing

# NSFLSs = "standard" or "cen-NS" or "sim-NS" or "sub-NS"
def main(antecedent_number=7, predict_past_points=9, noise_est_past_points=9, NSFLSs="standard", technique="ADONiS"):
    
    noise_free_train = pd.read_csv("timeser/lorenz_train.csv", header=None)[1]
    nf_sigma = pd.Series.std(noise_free_train)
        #Generating rules from noise free set
    train_obj = wang_mendel(noise_free_train, antecedent_number, predict_past_points)

    test_obj = testing(train_obj, noise_free_train) 
    
    test_obj.generate_firing_strengths(NSFLSs, technique, nf_sigma, noise_est_past_points)
    
    result = test_obj.apply_rules_to_inputs()
    
    
    print("The next point will be around: ",result)
    

"""
if __name__ == "__main__":
    print(" ")
    main(str(sys.argv[1]),sys.argv[2],sys.argv[3])
    """