import numpy as np
from tools.functions import *
import matplotlib.pyplot as plt
from tools.common_parameters import *



# parameters
method = "stepwise"             # method for generating the prototypes
percent_on = 0.1                # percentage of bits that are flipped
stimulus_length = 200      # length of the pattern
n_prototypes = n_classes        # number of prototypes
n_per_prototype = 10          # number of variations for subclasses per prototype
n_per_subclass= 20         # number of variations per subclass
plot_prototype = 4              # index of the prototype to plot
flip_first_round = 0.1     # percentage of bits that are flipped in the replication
flip_second_round = 0.05     # percentage of bits that are flipped in the replication



# parameters
print("This script generates data with the 'random' method by generating random prototypes in the first step and varying them in the second and third.")
print("Method:                                      ", method)
print("Number of classes:                           ", n_classes)
print("Number of input neurons (stimulus length):   ", stimulus_length)
print("Number of hidden neurons:                    ", N_hidden)
print("Number of class neurons:                     ", N_class)
print("Number of class neurons per class:           ", n_c_unit)


# generate data
prototypes = generate_prototypes(n_prototypes, percent_on, stimulus_length)
prototype_variations = generate_prototype_variations(prototypes, n_per_prototype, flip_first_round, inlcude_original=False)
final_variations = generate_prototype_variations(prototype_variations, n_per_subclass, flip_second_round)

# Does not work with include_original=True!!

# visualize the data

""" subset = np.append(prototypes[plot_prototype], prototype_variations[0][(plot_prototype * n_per_prototype):((plot_prototype * n_per_prototype) - 1 + n_per_prototype)]).reshape(n_per_prototype, stimulus_length)
plt.imshow(subset[0:100,:], cmap='gray')
plt.show()

imshow(final_variations[0][:550,:], cmap='gray')

final_variations[1].count(0)
prototype_variations[0].shape """


# split data into train and test
train_test_sequence_data = train_test_split(final_variations[0], final_variations[1], train_perccentage = 0.8, seed = 0)

# save data to file
save_data(train_test_sequence_data, 
          unique="finnegan" + "_" + str(flip_first_round) + "_" + str(flip_second_round) + "_" + str(n_classes) + "_" + str(stimulus_length) + "_" + str(len(final_variations[0])), 
          path='scripts/data/')
