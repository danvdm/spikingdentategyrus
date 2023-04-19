from .generate_data import *

n_classes = 5                          # number of classes
N_v = N_inputs = 100                     # number of input neurons
N_c = N_class = 20                      # number of class neurons
N_h = N_hidden = 50                     # number of hidden neurons

n_c_unit =  N_c/n_classes               # number of class neurons per class

# neuron equations

eqs_str_lif_wnrd = '''
dv/dt = (-g_leak*v + i_inj + I_rec + wnsigma*xi + I_d)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
age : 1
'''

eqs_str_lif_wnr = '''
dv/dt = (-g_leak*v + i_inj + I_rec + wnsigma*xi)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
age : 1
'''