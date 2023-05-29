# Parameters for the simulation
from brian2 import *
from tools.functions import *

n_classes = 0                          # number of classes
N_v = N_inputs = 100                     # number of input neurons
N_c = N_class = 0                      # number of class neurons
N_h = N_hidden = 500                     # number of hidden neurons

n_c_unit = 0 # N_c/n_classes               # number of class neurons per class

# neuron equations:
eqs_str_lif_wnrd = '''
dv/dt = (-g_leak*v + i_inj + I_rec + wnsigma*xi + I_d)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
age : 1
'''
eqs_str_lif_wnr = '''
dv/dt = (-g_leak*4*v + i_inj + I_rec + wnsigma*xi)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
q : 1
age : 1
'''
# *((-age+2)/2+1)
#  

dcmt = 60                              # duration of cycle
generations = 2
#sim_time = 2 # originally 300s

steepness = 5.0
perc_vis = 1
perc_hid = 0.5
age_v = np.concatenate((np.array(generate_pattern(N_v, perc_active = perc_vis)), np.repeat(1, N_c)))
#age_h = np.array(generate_pattern(N_h, perc_active = perc_hid))
age_h = np.random.uniform(-(generations+1), 1, N_h)


method = "euler"          # integration method for differential equations

#----------------------------------------- Neuron parameters
t_ref = 0.004 * second                  # refractory period 
#t_sim = dcmt*t_ref*sim_time                  # simulation time - originally 10000
bias_input_rate = 1000. * Hz            # Mean firing rate of bias Poisson spike train
beta_parameter = 2.04371561e+09 # * 1/amp 
gamma = np.exp(9.08343441e+00)* Hz      # Baseline firing rate
tau_noise = .001 * ms                   # noise time constant ?
tau_rec = t_ref                         
theta = .1 * volt  
Cm = 1e-12 * farad                      # membrane capacitance 
beta_fi = 1./cm/theta                   # beta for F-I curve  ??
sigma = 1.e-9                           # noise amplitude - This is now used in the queation creations instead of wnsigma --> not sure what it originally was for...
cal_i_lk = 0.0e-10                      # leak current
g_leak = 1e-9 * siemens                 # leak conductance
#dt = 0.00005 * second                 # time step --> Not necessary anymore ?!?!?!
# n_samples = ceil(t_sim/(dcmt*t_ref)+1)  # number of samples
wnsigma = 4.24e-11 * amp / second**-0.5 # This the reason why its slow. old: 4.24e-11

t_burn_percent = 10.                    # percentage of burn-in time
tau_learn = t_ref                       

deltaT = ((0.49-t_burn_percent/100)*dcmt*t_ref)

eta = 32e-3                             # learning rate 
epsilon = eta/beta_parameter*t_ref**2                   #--- not used apparently?!?!
epsilon_bias = eta/beta_parameter*t_ref*(1./bias_input_rate)  #--- not used apparently?!?!

deltaA  = eta/beta_parameter/tau_learn/deltaT*t_ref**2/2
deltaAbias = eta/beta_parameter/tau_learn/deltaT*t_ref*(1./bias_input_rate)/2 

i_inj = (- np.log(float(gamma))
         - np.log(float(t_ref))
         )/beta_parameter * amp                   # injected current ?
