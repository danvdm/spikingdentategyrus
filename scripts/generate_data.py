import numpy as np
from tools.functions import *
import matplotlib.pyplot as plt
from tools.common_parameters import *

# parameters
print("Number of classes:                           ", n_classes)
print("Number of input neurons (stimulus length):   ", N_inputs)
print("Number of hidden neurons:                    ", N_hidden)
print("Number of class neurons:                     ", N_class)
print("Number of class neurons per class:           ", n_c_unit)

# parameters
variation_prototype = 0.1       # percentage of bits that are flipped
stimulus_length = N_inputs      # length of the pattern
n_prototypes = n_classes        # number of prototypes
n_per_prototype = 1000          # number of variations per prototype
plot_prototype = 4              # index of the prototype to plot
replication_variation = 0.1     # percentage of bits that are flipped in the replication

# generate data
prototypes = generate_prototypes(n_prototypes, variation_prototype, stimulus_length)
prototype_variations = generate_prototype_variations(prototypes, n_per_prototype, replication_variation)

# visualize the data

""" subset = np.append(prototypes[plot_prototype], prototype_variations[0][(plot_prototype * n_per_prototype):((plot_prototype * n_per_prototype) - 1 + n_per_prototype)]).reshape(n_per_prototype, stimulus_length)
plt.imshow(subset[0:100,:], cmap='gray')
plt.show() """

# split data into train and test
train_test_sequence_data = train_test_split(prototype_variations[0], prototype_variations[1], train_perccentage = 0.8, seed = 0)

# save data to file
save_data(train_test_sequence_data, 
          var_prot=variation_prototype, 
          repl_var=replication_variation,
          path='scripts/data/')