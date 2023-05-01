import numpy as np
from brian2 import *
from tools.functions import *
import matplotlib.pyplot as plt
from tools.parameters_main import *

def main(Whv, b_v, b_c, b_h, Id, t_sim, sim_time, dorun = True, monitors=True, mnist_data = None, performance_metrics = True, n_classes = 10):

    start_scope()
    b_init = np.concatenate([b_v, b_c, b_h])
    netobjs = []
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
            threshold = 'v>theta',  # removed *volt
            refractory = t_ref,
            reset = "v = 0*volt",    # changed to string
            method = method
            )


    netobjs += [neuron_group_rvisible, neuron_group_rhidden]
      
    #Bias group
    Bv = PoissonGroup(N_v+N_c, rates = bias_input_rate)     #Noise injection to v
    Bh = PoissonGroup(N_h, rates = bias_input_rate)         #Noise injection to h
    
    netobjs+=[Bv,Bh]
    
    #---------------------- Initialize State Variables
    neuron_group_rvisible.I_d = 0. * amp
    neuron_group_rvisible.age = age_v
    neuron_group_rhidden.age = age_h


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
                        w = w + g * Apre
                        lastupdate = t
                        ''', 
                   method = method
                        )
    Srs.connect()
    netobjs+=[Sbv,Sbh,Srs]
    
    M_rec = Whv/beta_parameter
    for i in range(M_rec.shape[0]):
        Srs.w[i,:] = M_rec[i,:]

    period = dcmt*t_ref
    mod = 100
    ev = Clock(period/mod)
    ev.add_attribute(name = "n")
    ev.add_attribute(name = "cycle")
    ev.add_attribute(name = "tmod")
    ev.add_attribute(name = "mod")
    ev.add_attribute(name = "period")
    ev.n = 0
    ev.cycle = 0
    ev.tmod = 0
    ev.mod = mod
    ev.period = period  

    timepoint = []
    growth_factor_list = []
        
    # Each epoch consists of a LTP phase during which the data is presented (construction), 
    # followed by a free- running LTD phase (reconstruction). The weights are updated asynchronously 
    # during the time interval in which the neural sampling proceeds.

    @network_operation(clock = ev)
    def g_update(when='after'):
        tmod, n = custom_step(ev, sim_time)
        timepoint.append(ev.cycle)
        growth_factor = gomperz_function((ev.cycle*2-1), steepness)
        growth_factor_list.append(growth_factor)

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
        
        neuron_group_rvisible.age = age_v

        

    netobjs += [g_update]
    
    if performance_metrics:
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
        Mv=SpikeMonitor(neuron_group_rvisible)
        Mv=SpikeMonitor(neuron_group_rvisible[:N_v])
        Mc=SpikeMonitor(neuron_group_rvisible[N_v:])
        Mvmem=StateMonitor(neuron_group_rvisible[N_v:], variables='v', record=True, )
        netobjs += [Mh, Mv, Mc, Mvmem]

    net = Network(netobjs)
    if dorun:
        import time
        tic = time.time()      
        net.run(t_sim)
        toc = time.time()-tic
        print(toc)
        
    return locals()

    



