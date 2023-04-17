# Parameters for the simulation
from brian2 import *
from tools.common_parameters import *

dcmt = 35                               # duty cyle in multiples of t_ref

sim_time = 150
steepness = 1.0

#----------------------------------------- Neuron parameters
t_ref = 0.004 * second                  # refractory period 
t_sim = dcmt*t_ref*sim_time                  # simulation time - originally 10000
bias_input_rate = 1000. * Hz            # Mean firing rate of bias Poisson spike train
beta_parameter = 2.04371561e+09 # 1/amp 
gamma = np.exp(9.08343441e+00)* Hz      # Baseline firing rate
tau_noise = .001 * ms                   # noise time constant ?
tau_rec = t_ref                         # Time constant ofrecurrent, and bias synapses ?
theta = .1 * volt  
Cm = 1e-12 * farad                      # membrane capacitance 
beta_fi = 1./cm/theta                   # beta for F-I curve  ??
sigma = 7.e-10 * amp    #1.e-9          # noise amplitude - This is now used in the queation creations instead of wnsigma --> not sure what it originally was for...
cal_i_lk = 0.0e-10                      # leak current
g_leak = 1e-9 * siemens                 # leak conductance
# dt = 0.00005 * second                 # time step --> Not necessary anymore ?!?!?!
n_samples = t_sim/(dcmt*t_ref)+1        # number of samples
wnsigma = 4.24e-11 * amp                # This the reason why its slow. Use sigma instead!!!

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

#defaultclock.dt = dt                   # Not necessary anymore

# Equation strings: 

eqs_str_lif_wnrd = '''
dv/dt = (-g_leak*v + i_inj + I_rec + sigma*xi*t_ref**0.5 + I_d)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
'''

eqs_str_lif_wnr = '''
dv/dt = (-g_leak*v + i_inj + I_rec + sigma*xi*t_ref**0.5)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
'''

method = "euler"          # integration method for differential equations