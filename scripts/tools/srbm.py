import numpy as np
from brian2 import *
from tools.functions import *
import matplotlib.pyplot as plt
from tools.parameters import *

def main(Whv, b_v, b_c, b_h, Id, t_sim, sim_time, leak_helper, p_target = 0.05, sparsity_cost = 0.5, dorun = True, 
         monitors=True, mnist_data = None, display = True, n_classes = 10, age_neurons = False, age_leak = False, 
         threshold_ageing_degree = 1, age_threshold = False, set_connectivity = False, generations = 1,
         connectivity_born = 0.5, connectivity_mature = 0.5, turnover = False, prop_born = 0.5, apt_wt_str = 0.2, apt_age = 0.15, 
         apt_diff = 0.56, n_percent_apoptosis = 0.05, neurogenesis = False,
         gompertz = [0.8, 10]):

    start_scope()
    #b_init = np.concatenate([b_v, b_c, b_h])
    netobjs = []

    ageing_factor = generations/(sim_time*(dcmt*t_ref)/4/second) # 4 is the working domain of the gompertz function (see initialisation of age_h)
    print('ageing factor: ', ageing_factor)

    if age_neurons:
        if neurogenesis:
            age_h = np.concatenate((np.random.uniform(-1, 3, int(np.ceil(N_h * (1-prop_born)))), np.random.uniform(-4 * generations, -0.4 * generations, int(np.ceil(N_h*prop_born)))))
        else: 
            age_h = np.concatenate((np.random.uniform(-1, 3, int(np.ceil(N_h * (1-prop_born)))), np.repeat(-1000, int(np.ceil(N_h * prop_born)))))
        np.random.shuffle(age_h)
        if turnover:
            threshold_age = 'age > 3 + (apoptosis*1000)'
        else:
            threshold_age = 'age > 1000'
            
        if age_leak:
            eqs_hidden = eqs_str_lif_wnr_age
        else:
            eqs_hidden = eqs_str_lif_wnr
        if age_threshold:
            threshold_hidden = 'v>(((age - 1) * ' + str(threshold_ageing_degree) + ' * volt) + theta * (((-sign(age - 0.01) + 1) / 2) * 100))'
        else:
            threshold_hidden = 'v> (theta * (((-sign(age - 0.01) + 1) / 2) * 100))'
        if set_connectivity:
            connections_init = create_connection_matrix(N_input=N_v+N_c, N_hidden=N_h, probabilities=np.repeat(0.5, N_h), pmin = connectivity_born, pmax = connectivity_mature)
        else:
            connections_init = 1
    else: 
        age_h = np.repeat(1, N_h) 
        threshold_hidden = 'v>theta'
        connections_init = 1 
        eqs_hidden = eqs_str_lif_wnr
        threshold_age = 'age > 1000'

    
    
    
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
            model = eqs_hidden, # changed from eqs_h to eqs_str_lif_wnr
            threshold =  threshold_hidden,       # threshold_hidden,  # removed *volt 
            refractory = t_ref,
            reset = "v = 0*volt",    # changed to string
            method = method
            )
    
    eqs_helper_age = '''
                    dage/dt = age_factor/t_helper :1
                    t_helper : second
                    age_factor : 1
                    helper_leak : 1
                    apoptosis : 1'''

    eqs_helper_av_act = '''
                        dq/dt = ((-helper_leak*q)/t_helper) :1
                        t_helper : second
                        helper_leak : 1
                        n_hidden : 1'''
    
    neuron_group_rhidden_helper_age = NeuronGroup(\
            N_h,
            model = eqs_helper_age, 
            threshold = threshold_age,
            reset = 'age = -2',
            ) 
    
    neuron_group_rhidden_helper_av_act = NeuronGroup(\
        1,
        model = eqs_helper_av_act
        ) 

    netobjs += [neuron_group_rvisible, neuron_group_rhidden, 
                neuron_group_rhidden_helper_age,
                neuron_group_rhidden_helper_av_act]
      
    #Bias group
    Bv = PoissonGroup(N_v+N_c, rates = bias_input_rate)     #Noise injection to v
    Bh = PoissonGroup(N_h, rates = bias_input_rate)         #Noise injection to h
    
    netobjs+=[Bv,Bh]
    
    #---------------------- Initialize State Variables
    neuron_group_rvisible.I_d = 0. * amp
    
    neuron_group_rhidden_helper_age.t_helper = 1 * second
    neuron_group_rhidden_helper_age.helper_leak = leak_helper
    neuron_group_rhidden_helper_age.age = age_h
    neuron_group_rhidden_helper_age.apoptosis = np.repeat(1, N_h)
    neuron_group_rhidden_helper_age.age_factor = ageing_factor
    neuron_group_rhidden_helper_av_act.t_helper = 1 * second
    neuron_group_rhidden_helper_av_act.helper_leak = leak_helper
    neuron_group_rhidden_helper_av_act.n_hidden = N_h



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
                        w = w + g * Apost + ((g+1)/2*-q_post + (-g+1)/2*p) * cost * age_post
                        lastupdate = t
                        ''', 
                 on_post=''' 
                        Apre = Apre * exp((lastupdate-t)/tau_learn)
                        Apost = Apost * exp((lastupdate-t)/tau_learn)
                        Apost += deltaA
                        I_rec_pre += w * amp
                        w = w + g * Apre + ((g+1)/2*-q_post + (-g+1)/2*p) * cost * age_post
                        lastupdate = t
                        ''', 
                   method = method
                        )
    
    Srs.connect()

    # original:
    # * cost * age_post
    


    #  + ((((g+1)/2*-q_post) + ((-g+1)/2*p)) * cost * (-age_post+1))
    # * (-age_post+1))
    # (((g+1)/2)*-q_post + ((-g+1)/2)*p) * cost 


    Srs.p = p_target
    Srs.cost = sparsity_cost


    Shh_age = Synapses(neuron_group_rhidden, neuron_group_rhidden_helper_age)
    
    Shh_age.connect(j = "i")

    Shh_av_act = Synapses(neuron_group_rhidden, neuron_group_rhidden_helper_av_act,
                   on_pre='''q_post += 1e-10
                             ''')
    
    Shh_av_act.connect()
    
    

    netobjs+=[Sbv,Sbh,Srs, 
              Shh_age, Shh_av_act
              ]
    
    M_rec = Whv/beta_parameter
    
    #------------------------------------------ Connection matrix
    #connections = create_connection_matrix(N_input=N_v+N_c, N_hidden=N_h, probabilities=gomperz_function(neuron_group_rhidden.age, 2))

    M_rec = M_rec * connections_init #                                !!!!!!!!!!!!!!!
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
    connections = []
    av_wt_str = []
    differentiat = []
    check = []

    @network_operation(clock = ev)
    def g_update(when='after'):
        tmod, n = custom_step(ev, sim_time)            

        if age_neurons:
            neuron_group_rhidden.age = gomperz_function(neuron_group_rhidden_helper_age.age +gompertz[0], gompertz[1])
            if set_connectivity:
                Wts = np.full((N_v + N_c, N_h), np.nan)
                Wts[Srs.i[:], Srs.j[:]] = Srs.w[:]
                # apoptosis
                average_wt_strength = normalizer(abs(np.mean(Wts, axis= 0)))
                differentiation = normalizer(np.std(Wts, axis= 0))
                p_apoptosis = (apt_wt_str * (average_wt_strength) + apt_diff * (differentiation) + apt_age * (1-neuron_group_rhidden.age)) / (apt_wt_str + apt_diff + apt_wt_str)
                apoptosis = 1-(p_apoptosis <= p_apoptosis[np.argsort(p_apoptosis)[int(len(p_apoptosis)*n_percent_apoptosis)-1:int(len(p_apoptosis)*n_percent_apoptosis)][0]]) * 1
                neuron_group_rhidden_helper_age.apoptosis = apoptosis
                # weight update
                old_connections = ev.connections.copy()
                new_born_init = create_weight_matrix(N_v, N_h, N_c, sigma = 0.1) / beta_parameter
                ev.connections = update_connection_matrix(ev.connections, probabilities=neuron_group_rhidden.age, pmin = connectivity_born, pmax = connectivity_mature)
                
                random_new = ((ev.connections-old_connections > 0) * ev.connections) * new_born_init
                check.append(ev.connections-old_connections)
                Wts_updated = Wts * ev.connections + random_new
                Srs.w[:] = Wts_updated.flatten()


        neuron_group_rhidden.q = neuron_group_rhidden_helper_av_act.q / N_h

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
            if set_connectivity:
                weights.append(Wts_updated.flatten())
                connections.append(ev.connections.flatten())
                av_wt_str.append(average_wt_strength)
                differentiat.append(differentiation)
            #print(neuron_group_rhidden_helper_age.age)
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
        sMhh_age=StateMonitor(neuron_group_rhidden_helper_age, variables='age', record=True)
        sMhh_av_act=StateMonitor(neuron_group_rhidden_helper_av_act, variables='q', record=True)
        sMh_age=StateMonitor(neuron_group_rhidden, variables='age', record=True)
        sMh_v=StateMonitor(neuron_group_rhidden, variables='v', record=True)
        sMh_q=StateMonitor(neuron_group_rhidden, variables='q', record=True)
        sMhh_age=StateMonitor(neuron_group_rhidden_helper_age, variables='age', record=True)
        if N_c > 0:
            Mc=SpikeMonitor(neuron_group_rvisible[N_v:])
            Mvmem=StateMonitor(neuron_group_rvisible[N_v:], variables='v', record=True, )
            netobjs += [Mh, Mv, Mc, Mvmem, sMh_age, sMh_v, sMh_q,
                        sMhh_age, sMh_age, sMhh_age, sMhh_av_act
                        ]
        else: 
            Mc = None
            Mvmem = None
            netobjs += [Mh, Mv, sMh_age, sMh_v, 
                        sMhh_age, sMhh_av_act, sMh_q
                        ]
        
        

    net = Network(netobjs)
    if dorun:
        import time
        tic = time.time()      
        net.run(t_sim, report='text')
        toc = time.time()-tic
        print(toc)
        
    return locals()

    



