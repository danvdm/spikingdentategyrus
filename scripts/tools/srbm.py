import numpy as np
from brian2 import *
from tools.functions import *
import matplotlib.pyplot as plt
from tools.parameters import *

def main(Whv, b_v, b_c, b_h, Id, t_sim, sim_time, leak_helper, p_target = 0.05, sparsity_cost = 0.5, dorun = True, 
         monitors=True, mnist_data = None, display = True, n_classes = 10, age_neurons = False, generations = 1):

    start_scope()
    b_init = np.concatenate([b_v, b_c, b_h])
    netobjs = []

    qb = 4

    if age_neurons:
        age_h = np.random.uniform(-1, 3, N_h)
        threshold_hidden = 'v>((exp(-exp(-2*age)) + 0.0) * 10 * theta)'
        connections_init = create_connection_matrix(N_input=N_v+N_c, N_hidden=N_h, probabilities=np.repeat(0.5, N_h))
    else: 
        age_h = np.repeat(1, N_h)
        threshold_hidden = 'v>theta'
        connections_init = 1

    ageing_factor = generations/(sim_time*(dcmt*t_ref)/4/second) # 4 is the working domain of the gompertz function (see initialisation of age_h)
    print('ageing factor: ', ageing_factor)
    
    #------------------------------------------ Neuron Groups
    
    neuron_group_rvisible = NeuronGroup(\
            N_v+N_c,
            model = eqs_str_lif_wnrd, # changed from eqs_v to eqs_str_lif_wnrd
            threshold = 'v>theta',  # removed *volt
            refractory = t_ref,
            reset = "v = 0*volt",    # changed to string
            method = method
            )

    neuron_group_rhidden = NeuronGroup(\
            N_h,
            model = eqs_str_lif_wnr, # changed from eqs_h to eqs_str_lif_wnr
            threshold = threshold_hidden,  # removed *volt
            refractory = t_ref,
            reset = "v = 0*volt",    # changed to string
            method = method
            )
    
    eqs_helper = '''dq/dt = (-helper_leak*q)/t_helper :1
                    dage/dt = age_factor/t_helper :1
                    t_helper : second
                    age_factor : 1
                    helper_leak : 1'''

    neuron_group_rhidden_helper = NeuronGroup(\
            N_h,
            model = eqs_helper, 
            threshold = 'age > 3',
            reset = 'age = -1',
            )       

    netobjs += [neuron_group_rvisible, neuron_group_rhidden, neuron_group_rhidden_helper]
      
    #Bias group
    Bv = PoissonGroup(N_v+N_c, rates = bias_input_rate)     #Noise injection to v
    Bh = PoissonGroup(N_h, rates = bias_input_rate)         #Noise injection to h
    
    netobjs+=[Bv,Bh]
    
    #---------------------- Initialize State Variables
    neuron_group_rvisible.I_d = 0. * amp
    neuron_group_rhidden_helper.t_helper = 1 * second
    neuron_group_rhidden_helper.helper_leak = leak_helper
    neuron_group_rhidden_helper.age = age_h
    neuron_group_rhidden_helper.age_factor = ageing_factor


    #---------------------- Connections and Synapses
    #Bias units    
    Sbv = Synapses(Bv, neuron_group_rvisible, 
                   model='''
                        Apre : 1
                        Apost : 1
                        g : 1
                        w : 1
                        lastupdate : second
                        ''',
                   on_pre =''' 
                        Apre = Apre * exp((lastupdate-t)/tau_learn)
                        Apost = Apost * exp((lastupdate-t)/tau_learn)
                        Apre += deltaAbias
                        w = w + g * Apost
                        I_rec_post += w * amp
                        lastupdate = t''', 
                   on_post=''' 
                        Apre = Apre*exp((lastupdate-t)/tau_learn)
                        Apost = Apost*exp((lastupdate-t)/tau_learn)
                        Apost += deltaAbias
                        w=w+g*Apre
                        lastupdate = t
                        ''', 
                   method = method
                        )
    Sbv.connect(j='i')
    Sbv.w[:] = np.concatenate([b_v,b_c])/beta_parameter/bias_input_rate/tau_rec
    
    Sbh = Synapses(Bh, neuron_group_rhidden, 
                   model='''
                        Apre : 1
                        Apost : 1
                        g : 1
                        w : 1
                        lastupdate : second
                        ''', 
                   on_pre ='''
                        Apre=Apre*exp((lastupdate-t)/tau_learn)
                        Apost=Apost*exp((lastupdate-t)/tau_learn)
                        Apre+=deltaAbias
                        w=w+g*Apost
                        I_rec_post+= w * amp
                        lastupdate = t
                        ''', 
                   on_post='''
                        Apre=Apre*exp((lastupdate-t)/tau_learn)
                        Apost=Apost*exp((lastupdate-t)/tau_learn)
                        Apost+=deltaAbias
                        w=w+g*Apre
                        lastupdate = t
                        ''', 
                   method = method
                        )
    Sbh.connect(j='i')
    Sbh.w[:] = b_h[:]/beta_parameter/bias_input_rate/tau_rec
    
    Srs=Synapses(neuron_group_rvisible, neuron_group_rhidden,
                 model='''   
                        Apre : 1
                        Apost : 1
                        g : 1
                        w : 1
                        p : 1
                        cost : 1
                        lastupdate : second
                        ''', 
                 on_pre =''' 
                        Apre = Apre*exp((lastupdate-t)/tau_learn)
                        Apost = Apost*exp((lastupdate-t)/tau_learn)
                        Apre += deltaA
                        I_rec_post += w * amp
                        w = w + g * Apost 
                        lastupdate = t
                        ''', 
                 on_post=''' 
                        Apre = Apre * exp((lastupdate-t)/tau_learn)
                        Apost = Apost * exp((lastupdate-t)/tau_learn)
                        Apost += deltaA
                        I_rec_pre += w * amp
                        w = w + g * Apost 
                        lastupdate = t
                        ''', 
                   method = method
                        )
    
    Srs.connect()

    #  + ((((g+1)/2*-q_post) + ((-g+1)/2*p)) * cost * (-age_post+1))
    Srs.p = p_target
    Srs.cost = sparsity_cost


    Shh = Synapses(neuron_group_rhidden, neuron_group_rhidden_helper,
                   on_pre='''q_post += 0.0001
                             ''')
    Shh.connect(j='i')

    netobjs+=[Sbv,Sbh,Srs, Shh]
    
    M_rec = Whv/beta_parameter
    """ for i in range(M_rec.shape[0]):
        Srs.w[i,:] = M_rec[i,:] """
    
    #------------------------------------------ Connection matrix
    #connections = create_connection_matrix(N_input=N_v+N_c, N_hidden=N_h, probabilities=gomperz_function(neuron_group_rhidden.age, 2))

    M_rec = M_rec * connections_init
    Srs.w[:] = M_rec.flatten() # deeeegaaaaaaaaa!!!!!!!!!!!!!!!

    period = dcmt*t_ref
    mod = 100
    ev = Clock(period/mod)
    ev.add_attribute(name = "n")
    ev.add_attribute(name = "cycle")
    ev.add_attribute(name = "tmod")
    ev.add_attribute(name = "mod")
    ev.add_attribute(name = "period")
    ev.add_attribute(name = "connections")
    ev.n = 0
    ev.cycle = 0
    ev.tmod = 0
    ev.mod = mod
    ev.period = period  
    ev.connections = connections_init

        
    # Each epoch consists of a LTP phase during which the data is presented (construction), 
    # followed by a free- running LTD phase (reconstruction). The weights are updated asynchronously 
    # during the time interval in which the neural sampling proceeds.

    weights = []

    @network_operation(clock = ev)
    def g_update(when='after'):
        tmod, n = custom_step(ev, sim_time)            

        if age_neurons:
            neuron_group_rhidden.age = neuron_group_rhidden_helper.age 
            Wts = np.full((N_v + N_c, N_h), np.nan)
            Wts[Srs.i[:], Srs.j[:]] = Srs.w[:]
            ev.connections = update_connection_matrix(ev.connections, probabilities=gomperz_function(neuron_group_rhidden.age, 2))
            Wts_updated = Wts * ev.connections
            Srs.w[:] = Wts_updated.flatten()

        neuron_group_rhidden.q = neuron_group_rhidden_helper.q

        if tmod < 50:   # while below 50 cycles, clamp data to visible units. Otherwise input current = 0 below 50 is data phase, above 50 is reconstruction phase
            neuron_group_rvisible.I_d = Id[n] * amp
        else:
            neuron_group_rvisible.I_d = 0. * amp

        if tmod<=int(t_burn_percent): # while time is within burn in timeframe, set g = 0 (no stdp takes place)
            Srs.g = 0. # synapses connecting hidden and visible 
            Sbh.g = Sbv.g = 0. # synapses connecting biases to hidden/ visible layers 

        elif int(t_burn_percent)<=tmod<49: # if time is higher than burn in but lower than 50 cycles: g = 1, meaning 
            g_up = 1. 
            Srs.g = Sbv.g = Sbh.g =  g_up 
            
        elif 49<=tmod < 50+int(t_burn_percent):
            Srs.g = Sbv.g = Sbh.g = +0.
            
        elif 50+int(t_burn_percent) <= tmod <99:
            g_down = -1.
            Srs.g = Sbv.g = Sbh.g = g_down
            
        elif 99<= tmod:
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.

        if tmod==50:
            #neuron_group_rvisible.I_DATA=0
            Srs.Apre=0
            Srs.Apost=0
            Sbv.Apre=0
            Sbv.Apost=0
            Sbh.Apre=0
            Sbh.Apost=0
            #weights.append(Wts_updated.flatten())
        neuron_group_rvisible.age = age_v

    netobjs += [g_update]
    
    if display:
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = mnist_data
        res_hist_test=[]
        res_hist_train=[]
        test_data = test_iv
        test_labels = test_iv_l
        train_data = train_iv[:200]
        train_labels = train_iv_l[:200]
        plot_every = 10

        @network_operation(clock=Clock(dt=plot_every*dcmt*t_ref))
        def plot_performance(when='after'):   
            W = np.array(Srs.w).reshape(N_v+N_c, N_h)*beta_parameter
            Wvh=W[:N_v,:]
            Wch=W[N_v:,:] 
            accuracy_test = classification_free_energy(Wvh, Wch, b_h, b_c, test_data, test_labels, n_c_unit, n_classes=n_classes)[0]    
            res_hist_test.append(accuracy_test)
            
            accuracy_train = classification_free_energy(Wvh, Wch, b_h, b_c, train_data, train_labels, n_c_unit, n_classes=n_classes)[0]
            res_hist_train.append(accuracy_train)

            print("Train accuracy:", accuracy_train)
            print("Test accuracy:", accuracy_test)

        netobjs += [plot_performance]
    
    #--------------------------- Monitors
    if monitors:
        Mh=SpikeMonitor(neuron_group_rhidden)
        #Mv=SpikeMonitor(neuron_group_rvisible)
        Mv=SpikeMonitor(neuron_group_rvisible[:N_v])
        sMhh=StateMonitor(neuron_group_rhidden_helper, variables='q', record=True)
        sMh=StateMonitor(neuron_group_rhidden_helper, variables='q', record=True)
        sMh_age=StateMonitor(neuron_group_rhidden, variables='age', record=True)
        sMh_v=StateMonitor(neuron_group_rhidden, variables='v', record=True)
        sMhh_age=StateMonitor(neuron_group_rhidden_helper, variables='age', record=True)
        if N_c > 0:
            Mc=SpikeMonitor(neuron_group_rvisible[N_v:])
            Mvmem=StateMonitor(neuron_group_rvisible[N_v:], variables='v', record=True, )
            netobjs += [Mh, Mv, Mc, Mvmem, sMhh, sMh, sMh_age, sMh_v, sMhh_age]
        else: 
            Mc = None
            Mvmem = None
            netobjs += [Mh, Mv, sMhh, sMh, sMh_age, sMh_v, sMhh_age]
        
        

    net = Network(netobjs)
    if dorun:
        import time
        tic = time.time()      
        net.run(t_sim, report='text')
        toc = time.time()-tic
        print(toc)
        
    return locals()

    



