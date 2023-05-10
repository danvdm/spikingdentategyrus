import numpy as np
import matplotlib.pyplot as plt
from tools.functions import *
number_of_patterns = 100      # number of initial unique patterns of which for each pattern 9 variations are generated
length_v = 200                # length of the inidivual patterns - should be length of visible layer

# Generates data with the method used by Kim et al. (2023)

data = generate_data_kim(number_of_patterns, length_v)
# Some checks that it looks alright
""" for i in range(10):
    print(round(cosine_similarity(data[0,0,:], data[0,i,:]), 10))

plt.imshow(data[0,10:None:-1,:], cmap='gray') 
plt.show """

save_data(data, 'kim' + "_" + str(length_v) + "_" + str(number_of_patterns), path='scripts/data/', dtype='float32')
