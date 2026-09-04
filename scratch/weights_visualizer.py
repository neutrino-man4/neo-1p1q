import helpers.utils as ut
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plotting the weights')    
parser.add_argument('--seed', type=str, help='which run to load the weights for')
args=parser.parse_args()

checkpoint_dir=os.path.join('saved_models',args.seed,'checkpoints')

weights=ut.load_weights(checkpoint_dir)