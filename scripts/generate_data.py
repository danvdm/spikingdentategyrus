import numpy as np
from tools.functions import *
import matplotlib.pyplot as plt
from tools.parameters import *

save = False # save the data to file

# parameters
percent_on = 0.2                # percentage of bits that are flipped
stimulus_length = 100      # length of the pattern
n_seed_patterns = 5        # number of prototypes # can not be changed !!!!
n_prototype_per_seed = 10     # max 10!!          # number of variations for subclasses per prototype
n_variations_per_prototype= 10         # number of variations per subclass
plot_prototype = 4              # index of the prototype to plot
flip_first_round = 0.1     # percentage of bits that are flipped in the replication
flip_second_round = 0.05     # percentage of bits that are flipped in the replication


print("\nPlease run the data generation file if not done so yet!!")
print("\nNumber of seed patterns:                     ", n_seed_patterns)
print("Number of prototypes per seed patterns       ", n_prototype_per_seed)
print("Number of prototypes total:                  ", n_prototype_per_seed * n_seed_patterns)
print("Number of variations per prototype           ", n_variations_per_prototype) 
print("Number of patterns total                     ", n_prototype_per_seed * n_seed_patterns * n_variations_per_prototype) 
print("Number of input neurons (stimulus length):   ", stimulus_length)
print("Number of hidden neurons:                    ", N_hidden)
print("Number of class neurons:                     ", N_class)
print("Number of class neurons per class:           ", n_c_unit)



# generate data
#prototypes = generate_prototypes(n_seed_patterns, percent_on, stimulus_length)

## Weird way to find seed patterns that have a more or less uniform overlap matrix. Works only for 5 patterns now.

i = 0
matched = False
m = []
while i < 10000 and matched == False:

    p1 = generate_pattern(stimulus_length, percent_on)
    
    #Use this for less overlap 
    """ p2 = generate_overlapping_pattern(p1, 0.9)
    p3 = generate_overlapping_pattern(p1, 0.5)
    p4 = generate_overlapping_pattern(p1, 0.3)
    p5 = generate_overlapping_pattern(p1, 0.01) """

    # the results were generated with this
    p2 = generate_overlapping_pattern(p1, 0.7)
    p3 = generate_overlapping_pattern(p1, 0.6)
    p4 = generate_overlapping_pattern(p1, 0.5)
    p5 = generate_overlapping_pattern(p1, 0.4)

    percent_match = np.zeros((5, 5))
    percent_match[1, 0] = percent_overlap(p2, p1)
    percent_match[2, 0] = percent_overlap(p3, p1)
    percent_match[3, 0] = percent_overlap(p4, p1)
    percent_match[4, 0] = percent_overlap(p5, p1)
    percent_match[2, 1] = percent_overlap(p3, p2)
    percent_match[3, 1] = percent_overlap(p4, p2)
    percent_match[4, 1] = percent_overlap(p5, p2)
    percent_match[3, 2] = percent_overlap(p4, p3)
    percent_match[4, 2] = percent_overlap(p5, p3)
    percent_match[4, 3] = percent_overlap(p5, p4)

    if percent_match[4, 3] < 0.03 and len(unique(percent_match.flatten())) == 9 :
        matched = True
        print("Matched")
    i += 1

print(percent_match)
plt.hist(percent_match.flatten()[percent_match.flatten() > 0], bins=20)
plt.show()

prototypes = np.array([p1, p2, p3, p4, p5])

prototype_variations = generate_prototype_variations(prototypes, n_prototype_per_seed, flip_first_round)
final_variations = generate_final_variations(prototype_variations, n_variations_per_prototype, flip_second_round)

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

if save:
    save_data(final_variations, 
            unique="uniform_variations_100", 
            path='scripts/training_data/')
