import numpy as np
from brian2 import *
from tools.functions import *
from tools.parameters_reconstruction import *
import matplotlib.pyplot as plt

def main(Whv, b_v, b_c, b_h, Id, t_sim, dorun = True, monitors=True, display=False, mnist_data = None):
   
    start_scope()
    netobjs = []

    #------------------------------------------ Neuron Groups
    print("Creating equation")
    
    eqs_v = Equations(eqs_str_lif_wnrd, 
            Cm = 1e-12*farad,
            I_inj = i_inj,
            g = g_leak,
            sigma = sigma,                      # This was set to wnsigma in the original code
            tau_rec = tau_rec)
                                               
    eqs_h = Equations(eqs_str_lif_wnr, 
            Cm = 1e-12*farad,
            I_inj = i_inj,
            g = g_leak,
            sigma = sigma,                      # This was set to wnsigma in the original code
            tau_rec = tau_rec)

    neuron_group_rvisible = NeuronGroup(\
            N_v+N_c,                           
            model = eqs_v,
            threshold = 'v>theta', # removed: *volt
            refractory = t_ref,
            reset = "v = 0 * volt"
            )
    
    neuron_group_rhidden = NeuronGroup(\
            N_h,                                
            model = eqs_h,
            threshold = 'v>theta', # removed: *volt
            refractory = t_ref,
            reset = "v = 0*volt"
            )

    netobjs += [neuron_group_rvisible, neuron_group_rhidden]
   
    
    #Bias group
    Bv = PoissonGroup(N_v+N_c, rates = bias_input_rate) #Noise injection to h
    Bh = PoissonGroup(N_h, rates = bias_input_rate) #Noise injection to h
    
    netobjs+=[Bv,Bh]
    
    #---------------------- Initialize State Variables
    neuron_group_rvisible.I_d = 0.
    
    
    #---------------------- Connections and Synapses
    #Bias units    
    Sbv = Synapses(Bv, neuron_group_rvisible, 
              model='''
                Apre : 1
                Apost : 1
                g : 1
                w : 1
                lastupdate : second''', 
            on_pre =''' Apre = Apre * exp((lastupdate-t)/tau_learn)
                        Apost = Apost * exp((lastupdate-t)/tau_learn)
                        Apre += deltaAbias
                        w = w + g * Apost
                        I_rec_post += w * amp
                        lastupdate = t''', 
            on_post=''' Apre = Apre*exp((lastupdate-t)/tau_learn)
                        Apost = Apost*exp((lastupdate-t)/tau_learn)
                        Apost += deltaAbias
                        w=w+g*Apre
                        lastupdate = t''' 
                        )
    Sbv.connect(j='i')
    Sbv.w[:] = np.concatenate([b_v,b_c])/beta_parameter/bias_input_rate/tau_rec
    
    Sbh = Synapses(Bh, neuron_group_rhidden, 
              model='''
                Apre : 1
                Apost : 1
                g : 1
                w : 1
                lastupdate : second''', 
            on_pre =''' Apre = Apre * exp((lastupdate-t)/tau_learn)
                        Apost = Apost * exp((lastupdate-t)/tau_learn)
                        Apre += deltaAbias
                        w = w + g * Apost
                        I_rec_post += w * amp
                        lastupdate = t''', 
            on_post=''' Apre = Apre*exp((lastupdate-t)/tau_learn)
                        Apost = Apost*exp((lastupdate-t)/tau_learn)
                        Apost += deltaAbias
                        w=w+g*Apre
                        lastupdate = t''' 
                        )
    Sbh.connect(j='i')
    Sbh.w[:] = b_h[:]/beta_parameter/bias_input_rate/tau_rec
    
    Srs=Synapses(neuron_group_rvisible, neuron_group_rhidden,
           model='''   Apre : 1
                Apost : 1
                g : 1
                w : 1
                lastupdate : second''', 
        on_pre =''' Apre = Apre*exp((lastupdate-t)/tau_learn)
                    Apost = Apost*exp((lastupdate-t)/tau_learn)
                    Apre += deltaA
                    I_rec_post += w * amp
                    w = w + g * Apost
                    lastupdate = t''', 
        on_post=''' Apre = Apre * exp((lastupdate-t)/tau_learn)
                    Apost = Apost * exp((lastupdate-t)/tau_learn)
                    Apost += deltaA
                    I_rec_pre += w * amp
                    w = w + g * Apre
                    lastupdate = t''' 
                )
    Srs.connect()
    
    M_rec = Whv/beta_parameter
    for i in range(M_rec.shape[0]):
        Srs.w[i,:] = M_rec[i,:]
    
    netobjs+= [Sbv,Sbh,Srs,]

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

    @network_operation(clock = ev)
    def g_update(when='after'):
        tmod, n = custom_step(ev, sim_time)
        if tmod < 50:
            neuron_group_rvisible.I_d = Id[n]  * amp
        else:
            neuron_group_rvisible.I_d = Id[n]  * amp
    
        if tmod<=int(t_burn_percent):
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.

        elif int(t_burn_percent)<=tmod<49:
            g_up = 1.
            Srs.g = Sbv.g = Sbh.g =  learning_rate_decay(n,n0=4000)*g_up
            
        elif 49<=tmod < 50+int(t_burn_percent):
            Srs.g = Sbv.g = Sbh.g = +0.
            
        elif 50+int(t_burn_percent) <= tmod <99:
            g_down = -1.
            Srs.g = Sbv.g = Sbh.g = learning_rate_decay(n,n0=4000)*g_down
            
        elif 99<= tmod:
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.
    
#        if tmod==50:
#            #neuron_group_rvisible.I_DATA=0
#            Srs.Afre=0
#            Srs.Afost=0
#            Sbv.Afre=0
#            Sbv.Afost=0
#            Sbh.Afre=0
#            Sbh.Afost=0

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
            accuracy_test = classification_free_energy(Wvh, Wch, b_h, b_c, test_data, test_labels, n_c_unit)[0]    
            res_hist_test.append(accuracy_test)
            
            accuracy_train = classification_free_energy(Wvh, Wch, b_h, b_c, train_data, train_labels, n_c_unit)[0]
            res_hist_train.append(accuracy_train)

            print("Train accuracy:", accuracy_train)
            print("Test accuracy:", accuracy_test)

        netobjs += [plot_performance]
    
    #--------------------------- Monitors
    if monitors:
        Mh=SpikeMonitor(neuron_group_rhidden)
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