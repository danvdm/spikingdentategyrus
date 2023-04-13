# Parameters demo: 
n_classes = 10
N_v = N_inputs =784
N_c = N_class = 40
N_h = N_hidden =500

n_c_unit =  N_c/n_classes

dcmt = 75 #duty cyle in multiples of t_ref

#----------------------------------------- Neuron parameters
t_ref = 0.004 * second

T = dcmt*t_ref
init_delay=dcmt*t_ref
delay=0*ms
T1_s = init_delay
T1_e = T+init_delay
T2_s = T+init_delay+delay
T2_e = 2*T+delay+init_delay
T3_s = 2*T+2*delay+ init_delay
T3_e = 3*T+2*delay+ init_delay

t_sim = T3_e

t_ref = 0.004 * second
bias_input_rate = 1000. * Hz # added Hz
beta = 2.04371561e+09
gamma = np.exp(9.08343441e+00) * Hz # added Hz
tau_noise = .001 * second
tau_rec = t_ref
theta = .1 * volt
cm = 1e-12 * farad
beta_fi = 1./cm/theta
sigma = 1.e-9 * amp
cal_i_lk = 0.0e-10
g_leak = 1e-9 * siemens
dt = 0.00005
n_samples = t_sim/(dcmt*t_ref)+1
wnsigma = 4.24e-11

t_burn_percent = 10.
tau_learn = 0.01 * second

deltaT = ((0.49-t_burn_percent/100)*dcmt*t_ref)

eta = 0e-3
epsilon = eta/beta*t_ref**2*(dcmt*t_ref)/deltaT
epsilon_bias = eta/beta*t_ref*(1./bias_input_rate)*(dcmt*t_ref)/deltaT

deltaA  = eta/beta/tau_learn*(dcmt*t_ref)/deltaT*t_ref**2 / second
deltaAbias = eta/beta/tau_learn*(dcmt*t_ref)/deltaT*t_ref*(1./bias_input_rate) / second

i_inj = (- np.log(float(gamma))
         - np.log(float(t_ref))
         )/beta * amp

sigm = lambda x: 1./(1+exp(-x))
#helper parameters

Bu = 3.2 * Hz
wb = 12.5
N_helper=100
         
         
def exp_prob_beta_gamma(dt, beta, g_leak, gamma, t_ref):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func

def learning_rate_decay(n,n0=1):
    return float(n0)/(float(n0)+n)

Th = Bu * wb
i_inj_v_helper =  Th/beta_fi
i_inj_v_helper

#----------------------------------------- Neuron equations

eqs_str_lif_wnrd = '''
dv/dt = (-g*v + I_inj + I_rec + sigma*xi*t_ref**0.5 + I_d)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
'''

eqs_str_lif_wnr = '''
dv/dt = (-g_leak*v + i_inj + I_rec + sigma*xi*t_ref**0.5)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
'''