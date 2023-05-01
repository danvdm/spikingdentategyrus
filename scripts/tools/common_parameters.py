n_classes = 0                          # number of classes
N_v = N_inputs = 200                     # number of input neurons
N_c = N_class = 0                      # number of class neurons
N_h = N_hidden = 50                     # number of hidden neurons

n_c_unit = 0 # N_c/n_classes               # number of class neurons per class

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