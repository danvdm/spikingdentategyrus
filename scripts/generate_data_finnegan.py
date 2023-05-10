import numpy as np
from tools.functions import *
import matplotlib.pyplot as plt
from tools.common_parameters import *



# parameters
method = "stepwise"             # method for generating the prototypes
percent_on = 0.1                # percentage of bits that are flipped
stimulus_length = 200      # length of the pattern
n_prototypes = 5        # number of prototypes
n_per_prototype = 10     # max 10!!          # number of variations for subclasses per prototype
n_per_subclass= 10         # number of variations per subclass
plot_prototype = 4              # index of the prototype to plot
flip_first_round = 0.1     # percentage of bits that are flipped in the replication
flip_second_round = 0.05     # percentage of bits that are flipped in the replication



print("\nPlease run the data generation file if not done so yet!!")
print("\nNumber of seed patterns:                     ", n_prototypes)
print("Number of prototypes per seed patterns       ", n_per_prototype)
print("Number of prototypes total:                  ", n_per_prototype * n_prototypes)
print("Number of variations per prototype           ", n_per_subclass) 
print("Number of patterns total                     ", n_per_prototype * n_prototypes * n_per_subclass) 
print("Number of input neurons (stimulus length):   ", stimulus_length)
print("Number of hidden neurons:                    ", N_hidden)
print("Number of class neurons:                     ", N_class)
print("Number of class neurons per class:           ", n_c_unit)



# generate data
prototypes = generate_prototypes(n_prototypes, percent_on, stimulus_length)
prototype_variations = generate_prototype_variations(prototypes, n_per_prototype, flip_first_round)
final_variations = generate_final_variations(prototype_variations, n_per_subclass, flip_second_round)

""" imshow(np.row_stack([prototypes[0], prototypes[0], prototypes[0], prototypes[0], prototypes[0], prototypes[0], prototypes[0], prototypes[0], prototypes[0], prototypes[0],
                     prototypes[1], prototypes[1], prototypes[1], prototypes[1], prototypes[1], prototypes[1], prototypes[1], prototypes[1], prototypes[1], prototypes[1], 
                     prototypes[2], prototypes[2], prototypes[2], prototypes[2], prototypes[2], prototypes[2], prototypes[2], prototypes[2], prototypes[2], prototypes[2], 
                     prototypes[3], prototypes[3], prototypes[3], prototypes[3], prototypes[3], prototypes[3], prototypes[3], prototypes[3], prototypes[3], prototypes[3],
                     prototypes[4], prototypes[4], prototypes[4], prototypes[4], prototypes[4], prototypes[4], prototypes[4], prototypes[4], prototypes[4], prototypes[4]]), 
                     cmap='gray')
plt.show()
imshow(prototype_variations[0], cmap='gray')
plt.show() """


# save data to file
save_data(final_variations, 
          unique="finnegan", 
          path='scripts/data/')
