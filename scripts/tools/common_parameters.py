n_classes = 10                          # number of classes
N_v = N_inputs =784                     # number of input neurons
N_c = N_class = 40                      # number of class neurons
N_h = N_hidden =50      #500           # number of hidden neurons

n_c_unit =  N_c/n_classes               # number of class neurons per class

# neuron equations

eqs_str_lif_wnrd = '''
dv/dt = (-g*v + I_inj + I_rec + sigma*xi*t_ref**0.5 + I_d)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
'''

eqs_str_lif_wnr = '''
dv/dt = (-g_leak*v + i_inj + I_rec + sigma*xi*t_ref**0.5)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
'''