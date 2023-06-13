import numpy as np
from brian2 import *
import os

# Set path to data
path_to_data = 'data/mnist_reduced.pkl.gz'

def exp_prob_beta_gamma(dt, beta, g_leak, gamma, t_ref):
    '''Returns a function that takes a vector of membrane potentials and returns a vector of 0s and 1s'''
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func

def bound_data(data, min_p = 0.0001, max_p = .95, binary = False):
    '''Bounds the data to be between min_p and max_p. If binary, then the data is bounded to be between 0.5 and 0.5.'''
    if not binary:
        max_p_ = max_p
        min_p_ = min_p
    else:
        max_p_ = 0.5
        min_p_ = 0.5
    data[data >= max_p_] = max_p
    data[data < min_p_] = min_p
    
def select_equal_n_labels(n, data, labels, classes = None, seed=None):
    '''Selects n samples from data and labels, such that the number of samples from each class is equal.'''
    if classes is None:
        classes = range(10)    
    n_classes = len(classes)
    n_s = np.ceil(float(n)/n_classes)
    max_i = [np.nonzero(labels==i)[0] for i in classes]
    if seed is not None:
        np.random.seed(seed)
    f = lambda x, n: np.random.randint(0, int(x)-1, int(n))
    a = np.concatenate([max_i[i][f(len(max_i[i]), n_s)] for i in classes])
    np.random.shuffle(a)
    iv_seq = data[a]
    iv_l_seq = labels[a]
    return iv_seq, iv_l_seq   

def get_data(n_samples, min_p = 0.0001, max_p = .95, binary = False, seed=None, datafile = path_to_data, num_classes = range(10), load_from_drive = True, data = None):
    '''Loads data either from drive or from memory. Returns the input vector sequence, the input vector label sequence, the training input vector, the training input vector label, the test input vector, and the test input vector label.'''
    if load_from_drive:
        import gzip, pickle
        mat = pickle.load(gzip.open(datafile, 'r'), encoding='latin1')
        train_iv = mat['train']
        train_iv_l = mat['train_label']
        test_iv = mat['test']
        test_iv_l = mat['test_label']
    else:
        train_iv = data[0]
        train_iv_l = data[1]
        test_iv = data[2]
        test_iv_l = data[3]
    
    bound_data(train_iv, min_p, max_p, binary)
    bound_data(test_iv, min_p, max_p, binary)
    
    iv_seq, iv_l_seq = select_equal_n_labels(n_samples, train_iv, train_iv_l, seed = seed, classes=num_classes)
    
    return iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l

def prepare_data(n_samples = None, min_p = 1e-4, max_p=.95, binary=False, seed=None, n_classes = range(10)):
    '''Loads data and returns the input vector sequence, the input vector label sequence, the training input vector, the training input vector label, the test input vector, and the test input vector label.'''
    #------------------------------------------ Create Input Vector
    mnist_data = get_data(n_samples,
                            min_p = min_p,
                            max_p = max_p,
                            binary = binary,
                            seed = seed, 
                            num_classes = n_classes)
    iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = mnist_data
    return iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l

def clamped_input_transform(input_vector, min_p=1e-7, max_p=0.999):
    '''Transforms the input vector to be between min_p and max_p.'''
    s = np.array(input_vector) #Divide by t_ref to get firing rates
    s[s<min_p] = min_p
    s[s>max_p] = max_p
    s =  -np.log(-1+1./(s))
    return s

def create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = .00001, max_p = .95):
    '''Creates the input vector sequence with the clamped input.'''
    Id = np.ones([iv_seq.shape[0], iv_seq.shape[1]+N_c])*min_p
    
    for i in range(iv_seq.shape[0]):
        cl = np.zeros(N_c) 
        cl[int(iv_l_seq[i]*n_c_unit):int((iv_l_seq[i]+1)*n_c_unit)] = max_p
        Id[i,N_v:] = clamped_input_transform(cl, min_p = min_p, max_p = max_p)
        Id[i,:N_v] = clamped_input_transform(iv_seq[i,:], min_p = min_p, max_p = max_p)

    return Id

def create_Id(N_v, N_c, n_c_unit, beta, n_samples = None, data = True, c_min_p = 1e-4, c_max_p = .95, seed = None):
    '''Creates the input vector sequence with the clamped input. If data is True, then the MNIST data is loaded. 
       If data is a tuple, then the data is used. If data is False, then the input vector sequence is all zeros.'''
    if hasattr(data, '__len__'):
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
        Idp = create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = c_min_p, max_p = c_max_p)
        Id = (Idp /beta)
    elif data == True:
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = prepare_data(n_samples, seed = seed)
        Idp = create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = c_min_p, max_p = c_max_p)
        Id = (Idp /beta)
    else:
        Id = np.zeros([n_samples,N_v+N_c])  
    return Id

def classification_free_energy(Wvh, Wch, b_h, b_c, test_data, test_labels, n_c_unit, n_classes = 10):
    '''Calculates the classification free energy for the test data.'''
    numcases = len(test_labels);
    F = np.zeros([int(numcases), int(n_classes)]);
    for i in range(n_classes):
        X= np.zeros([int(numcases), int(n_c_unit)*int(n_classes)]);
        X[:, int(n_c_unit*i):int(n_c_unit*(i+1))] = 1;
        F[:,i] = np.tile(b_c[i],numcases)*X[:,i]+\
                 np.sum(np.log(np.exp(np.dot(test_data, Wvh)+np.dot(X,Wch)+np.tile(b_h,numcases).reshape(numcases,-1))+1), axis=1)
    prediction= np.argmax(F, axis=1);
    accuracy = sum(prediction==test_labels)/numcases # changed from: 1-float(sum(prediction!=test_labels))/numcases
    assert 1>=accuracy>=.1/n_classes
    return accuracy, prediction==test_labels

""" def monitor_to_spikelist(Ms):
    if len(Ms.spikes)>0:
        s = np.array(Ms.spikes)
        id_list = range(len(Ms.source))
        s[:,1] = s[:,1] * 1000 #SpikeList takes ms
        return spikes.SpikeList(spikes = s, id_list = id_list)
    else:
        return spikes.SpikeList(id_list = id_list) """

def custom_step(clock_object):
    '''Custom step function for the clock object.'''
    tmod_now, n_now = clock_object.tmod, clock_object.n
    clock_object.tmod = np.mod(clock_object.tmod+1, clock_object.mod)
    clock_object.n = int(clock_object.t/(clock_object.period))
    return tmod_now, n_now

def create_bias_vectors(N_v, N_c, N_h):
    '''Creates the bias vectors for the RBM.'''
    bias_v = b_v = np.zeros(N_v)
    bias_h = b_h = np.zeros(N_h)
    bias_c = b_c = np.zeros(N_c)
    return b_v, b_c, b_h    

def create_weight_matrix(N_v, N_h, N_c, sigma = 0.1):
    '''Creates the weight matrix for the RBM.'''
    return np.random.normal(0, sigma, size=(N_v+N_c, N_h))

def create_rbm_parameters(N_v, N_h, N_c, wmean=0, b_vmean=0, b_hmean=0):
    '''Creates the RBM parameters.'''
    #------------------------------------------ Bias and weights
    b_v, b_c, b_h = create_bias_vectors(N_v, N_c, N_h)
    Whv = create_weight_matrix(N_v, N_h, N_c, sigma = 0.1)    
    Whv+= wmean
    b_v+= b_vmean
    b_h+= b_hmean
    return Whv, b_v, b_c, b_h

""" def visualise_connectivity(S):
    Ns = 20 # len(S.source)
    Nt = 20 # len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10) 
    plot(ones(Nt), arange(Nt), 'ok', ms=10) 
    for i, j in zip(S.i[0:20], S.j[0:20]):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1) """

def load_matrices(date, time, path = "output/"):
    '''Loads the matrices from the output folder.'''
    path = path+date+"/"+time+"/"
    try:
        W = np.load(path+"/W.dat", allow_pickle=True)
        Wvh = np.load(path+"/Wvh.dat", allow_pickle=True)
        Wch = np.load(path+"/Wch.dat", allow_pickle=True)
        mBv = np.load(path+"/mBv.dat", allow_pickle=True)
        mBh = np.load(path+"/mBh.dat", allow_pickle=True)
        b_c = np.load(path+"/b_c.dat", allow_pickle=True)
        b_v = np.load(path+"/b_v.dat", allow_pickle=True)
        b_h = np.load(path+"/b_h.dat", allow_pickle=True)
        mB = np.load(path+"/mB.dat", allow_pickle=True)
        print("Matrices loaded from output/"+path+"/")        
    except:
        print("File not found. Try again.")
        return None, None, None, None, None, None, None, None, None

    return W, Wvh, Wch, mBv, mBh, b_c, b_v, b_h, mB

def create_single_Id(idx, data, N_v, N_c, n_c_unit, beta_parameter, min_p = 1e-16, max_p = .9999, seed = None, mult_class=0.0, mult_data=1.0):
    iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
    Idp = np.ones([N_v+N_c])*min_p
    i = np.nonzero(iv_l_seq==idx)[0][0]
    cl = np.zeros(N_c)
    cl[int(iv_l_seq[i]*n_c_unit):int((iv_l_seq[i]+1)*n_c_unit)] = max_p
    Idp[N_v:] = clamped_input_transform(cl, min_p = min_p, max_p = max_p)*mult_class
    Idp[:N_v] = clamped_input_transform(iv_seq[i,:], min_p = min_p, max_p = max_p)*mult_data
    Id = (Idp /beta_parameter)
    return Id

def exp_prob_beta_gamma(dt, beta, g_leak, gamma, t_ref):
    '''Returns a function that calculates the probability of a spike given the membrane potential V.'''
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func
  
def custom_step(clock_object, sim_time):
    tmod_now, n_now = clock_object.tmod, clock_object.n
    clock_object.tmod = np.mod(clock_object.tmod+1, clock_object.mod)
    clock_object.n = int(clock_object.t/(clock_object.period))
    clock_object.cycle += (1 / (sim_time * 100)) 
    return tmod_now, n_now

def gomperz_function(x, steepness):
    return np.exp(-np.exp(-steepness*x))

def spike_histogram(spike_monitor, t_start, t_stop, s_per_s = True):
    '''
    Returns firing rate of spike_monitor between t_start and t_stop
    '''
    import numpy as np
    delta_t = t_stop - t_start
    if isinstance(spike_monitor, dict):
        k, v = (spike_monitor["k"], spike_monitor["v"])
    else: 
        k, v = zip(*spike_monitor.spike_trains().items())   
    def f(s):
        idx_low = s >= t_start
        idx_high = s < t_stop
        idx = idx_low * idx_high
        return np.sum(idx)
    if s_per_s == True:
        s_per_s = 1
    else:
        s_per_s = delta_t
    count = np.array(list(map(f, v)), dtype='float')/ delta_t * s_per_s # count spikes per neuron per second
    return np.array(list(zip(*[k,count])))

def save_matrices(W, Wvh, Wch, mBv, mBh, b_c, b_v, b_h, mB, date_str, date_time_str, path = "output/"):
    mypath = path+date_str+"/"+date_time_str[11:13]+"-"+date_time_str[14:16]
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    W.dump(mypath+"/W.dat")
    Wvh.dump(mypath+"/Wvh.dat")
    Wch.dump(mypath+"/Wch.dat")
    mBv.dump(mypath+"/mBv.dat")
    mBh.dump(mypath+"/mBh.dat")
    b_c.dump(mypath+"/b_c.dat")
    b_v.dump(mypath+"/b_v.dat")
    b_h.dump(mypath+"/b_h.dat")
    mB.dump(mypath+"/mB.dat")

    print("Matrices saved to output/"+mypath)

def generate_pattern(lenght, perc_active = 0.1):
    '''Generates a random pattern with perc_active active units.'''
    pattern = np.zeros(lenght)
    n_active = int(lenght*perc_active)
    active_idx = np.random.choice(lenght, n_active, replace=False)
    pattern[active_idx] = 1
    return pattern

def generate_prototypes(n_prototypes, p, length):
    prototypes = np.zeros((n_prototypes, length))
    for i in range(n_prototypes):
        prototypes[i] = generate_pattern(length, p)
    return prototypes

def train_test_split(data, labels, train_perccentage = 0.8, seed = 0):
    np.random.seed(seed)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    labels = np.array(labels)[idx]
    train_idx = int(len(data) * train_perccentage)
    train_data = data[:train_idx]
    train_labels = labels[:train_idx]
    test_data = data[train_idx:]
    test_labels = labels[train_idx:]
    return train_data, train_labels, test_data, test_labels

def save_data(data, unique, path = "data/", dtype = "object"):
    import pickle
    if not os.path.isdir(path):
        os.makedirs(path)
    file_name = path + "data_" + str(unique)+".pkl"
    pickle.dump(data, open(file_name,"wb"))


def load_data(unique, path = "data/"):
    import pickle
    file_name = path + str(unique) + ".pkl"
    try:
        data = pickle.load(open(file_name, "rb"))
        print("Data loaded from " +file_name)   
    except:
        print("File not found. Try again.")
        return None
    return data

def frequency_classification(spike_monitor, n_classes, n_neurons_per_class, t_ref=0.004, t_start=0.3, t_end=0.6, delay = 10, confidence = True):
    '''Makes a classification based on the frequency of the spikes in the spike_monitor.'''

    frequencies = np.array(spike_histogram(spike_monitor, np.asarray(t_start) * second + delay * np.asarray(t_ref) * second, np.asarray(t_end) * second)).T[1]
    reshaped_frequencies = frequencies.reshape(n_classes, int(n_neurons_per_class))
    if confidence:
        from scipy.stats import kruskal
        param_list = []
        for i in range(reshaped_frequencies.shape[0]):
            param_list.append(reshaped_frequencies[i])

        krusk = kruskal(*param_list)
        print("p-value: ", round(krusk[1], 3), ". Confidence: ", "High" if krusk[1] < 0.1 else "Low", sep='')

    return np.argmax(np.sum(reshaped_frequencies, axis=1), axis=0)

def create_timepoints(Ids, init_delay, delay, T):
    '''Creates timepoints for the input patterns and sets them as global variables.'''
    n_inputs = Ids.shape[0]-1
    t_s = 0 * second + init_delay
    t_e = 0 * second
    initial_delay = init_delay
    timepoints = []
    time_points_dict = {}
    for i in np.arange(n_inputs)+1:
        t_s = t_e + delay + initial_delay
        t_e = t_s + T
        name_t_s = "T" + str(i) + "_s"
        name_t_e = "T" + str(i) + "_e"
        globals()[name_t_s] = t_s
        globals()[name_t_e] = t_e
        initial_delay = 0
        timepoints.append(globals()[name_t_s])
        timepoints.append(globals()[name_t_e])
        time_points_dict[name_t_s] = t_s
        time_points_dict[name_t_e] = t_e
    sim_time = t_e
    return (timepoints, sim_time, time_points_dict)




def average_activation_degree(x, y, percent = True):
    '''Calculates the average activation degree of two patterns. If percent = True, returns the percentage of active neurons in the pattern.'''
    if len(x.shape) == 1:
        activation = (np.count_nonzero(x)+np.count_nonzero(y))/2
        len_pattern = x.shape[0]
    else:
        activation = (np.count_nonzero(x)/x.shape[0]+np.count_nonzero(y)/y.shape[0])/2
        len_pattern = x.shape[1]
    if percent: 
        return activation/len_pattern
    
def generate_overlapping_pattern(x, percent_overlap):
    '''Generates overlapping patterns as described in Kim et al. 2023.'''
    # get indices of active neurons in x
    idx_1 = np.where(x == 1)[0]
    idx_0 = np.where(x == 0)[0]
    num_overlap = int(len(idx_1)*percent_overlap)

    overlapping = np.random.choice(idx_1, num_overlap, replace=False)
    new = np.random.choice(idx_0, len(idx_1) - num_overlap, replace=False)

    out = np.zeros(len(x))
    out[overlapping] = 1
    out[new] = 1
    return out

def generate_data_kim(num_samples, len_pattern, percent_active = 0.1):
    '''Creates data with 10 different levels of overlap between patterns. [i,0,:] = original pattern, [i,j,:] = pattern with j/10 overlap
    The procedure is taken from the descriptions of Kim et al. 2023'''
    patterns = np.zeros((num_samples, 10, len_pattern))
    labels = np.zeros((num_samples, 10, 2))
    for i in range(num_samples):
        x = generate_pattern(len_pattern, perc_active = percent_active)
        patterns[i,0,:] = x
        labels[i, 0, :] = np.array([i, 0])
        for j in np.arange(9)+1:
            patterns[i, j,:] = generate_overlapping_pattern(x, round((j/10), 1))
            labels[i, j, :] = np.array([i, j])
    return patterns, labels


def cosine_similarity(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def create_Id_pattern(n, data, N_v, N_c, n_c_unit, beta_parameter, on_off_ratio, seed = 0, data_mult = 1, class_mult = 0, allowed_labels = None):
    np.random.seed(seed)
    Ids = [create_single_Id(0, data, N_v = N_v, N_c = N_c, n_c_unit = n_c_unit, 
                    beta_parameter = beta_parameter, 
                    mult_class = 0.0, mult_data = 0.0, 
                    min_p = 1e-4, max_p = .95)]
    labels_out = []
    if allowed_labels is None:
        labels = np.unique(data[1])
    else:
        labels = allowed_labels
    for i in range(n):
        label = np.random.choice(labels, 1)[0]
        Ids.append(create_single_Id(label, data, N_v = N_v, N_c = N_c, n_c_unit = n_c_unit, 
                    beta_parameter = beta_parameter, 
                    mult_class = class_mult, mult_data = data_mult, 
                    min_p = 1e-4, max_p = .95))
        labels_out.append(label)
        for j in range(on_off_ratio):
            Ids.append(create_single_Id(0, data, N_v = N_v, N_c = N_c, n_c_unit = n_c_unit, 
                    beta_parameter = beta_parameter, 
                    mult_class = 0.0, mult_data = 0.0, 
                    min_p = 1e-4, max_p = .95))
            labels_out.append(label)
    return (np.column_stack(Ids).T, labels_out)

def calculate_hamming_distance(x, y):
    return np.sum(np.abs(x-y))

def calculate_percent_match(a, b):
    return 1 - calculate_hamming_distance(a, b) / len(a)

def percent_overlap(pattern1, pattern2):
    return np.sum(pattern1*pattern2)/(len(pattern1) + len(pattern2))

def shufle_percent(original, percent_variation):
    # git indices of active bits
    active_bits = np.where(original == 1)[0]
    inactice_bits = np.where(original == 0)[0]
    # number of bits to flip
    n_flip = int(len(active_bits) * percent_variation)
    # get random indices to flip
    flip_indices = np.random.choice(active_bits, n_flip, replace=False)
    flip_inactive_indices = np.random.choice(inactice_bits, n_flip, replace=False)
    # flip bits
    new_pattern = np.copy(original)
    new_pattern[flip_indices] = 0
    new_pattern[flip_inactive_indices] = 1
    return new_pattern
    
def generate_prototype_variations(prototypes, n_sub, percent_variation):
    '''function to generate variations of prototypes with a certain percentage of flipped bits. Is used to generate the finnegan data'''
    n_prototypes, length = prototypes.shape
    variations = np.zeros((n_prototypes*(n_sub), length))
    original_prototypes = []
    for i in range(n_prototypes):
        for j in range(n_sub):
            variations[i*n_sub+j] = shufle_percent(prototypes[i], percent_variation)
            original_prototypes.append(i*10+j)
    return variations, original_prototypes

def generate_final_variations(prototype_variations, n_per_subclass, flip_second_round):
    prototypes, labels = prototype_variations
    n_prototypes, length = prototypes.shape
    variations = np.zeros((n_prototypes, n_per_subclass, length))
    original_prototypes = []
    for i in range(n_prototypes):
        for j in range(n_per_subclass):
            variations[i, j] = shufle_percent(prototypes[i], flip_second_round)
            original_prototypes.append(labels[i])
    return variations, original_prototypes

def train_test_split_finnegan(data_finnegan, split):
    """
    Splits the data into training and test data.
    """
    data_train = np.zeros((len(data_finnegan[0]), int(split*len(data_finnegan[0][0])), len(data_finnegan[0][0][0])))
    data_test = np.zeros((len(data_finnegan[0]), len(data_finnegan[0][0])-int(split*len(data_finnegan[0][0])), len(data_finnegan[0][0][0])))
    for i in range(len(data_finnegan[0])):
        data_train[i] = data_finnegan[0][i][:int(split*len(data_finnegan[0][i]))]
        data_test[i] = data_finnegan[0][i][int(split*len(data_finnegan[0][i])):]
    return data_train, data_test

def create_test_Id(test_data, off_time):
    Id = clamped_input_transform(test_data[0], min_p = 1e-4, max_p = .95)
    for h in range(off_time):
            Id = np.row_stack((Id, np.zeros((test_data.shape[1]))))
    for i in np.arange(test_data.shape[0]-1)+1:
        Id = np.row_stack((Id, clamped_input_transform(test_data[i], min_p = 1e-4, max_p = .95)))
        for j in range(off_time):
            Id = np.row_stack((Id, np.zeros((test_data.shape[1]))))
    return Id

def normalizer(x):
    return x/np.max(np.abs(x))

def hamming_distances_test(spike_monitor, test_show_times, time_points_dict, off_time=1, normalize = True, binarize = False, threshold = 10):
    hamming_distances = []
    percent_match_list = []
    originals = []
    recovereds = []
    for i in test_show_times:
        t_start_stimulus = time_points_dict["T"+ str(i)+"_s"]
        t_stop_stimulus = time_points_dict["T"+ str(i)+"_e"]
        t_start_recover = time_points_dict["T"+ str(i+1)+"_s"]
        t_stop_recover = time_points_dict["T"+ str(i+off_time)+"_e"]
        orig = spike_histogram(spike_monitor, t_start=t_start_stimulus, t_stop=t_stop_stimulus).T[1]
        recover = spike_histogram(spike_monitor, t_start=t_start_recover, t_stop=t_stop_recover).T[1]
        if normalize:
            orig = normalizer(orig)
            recover = normalizer(recover)
        if binarize:
            orig = np.where(orig > threshold, 1, 0)
            recover = np.where(recover > threshold, 1, 0)
        hamming_distances.append(calculate_hamming_distance(orig, recover))
        percent_match_list.append(calculate_percent_match(orig, recover))
        originals.append(orig)
        recovereds.append(recover)
    return hamming_distances, percent_match_list, originals, recovereds

def create_finnegan_Ids(train_test_data_finnegan, off_time = 1, n_per_prototype = 10, n_main_classes = 5):
    # n_per_prototype is the value that should not be above 10 in the parameters file

    num_prototypes = len(train_test_data_finnegan[0])
    n_per_prototype_train = len(train_test_data_finnegan[0][0])
    n_per_prototype_test = len(train_test_data_finnegan[1][0])
    num_vis = len(train_test_data_finnegan[0][0][0])

    lenght_out = n_per_prototype_train * num_prototypes + 2 * ((n_per_prototype_test + off_time * n_per_prototype_test) * num_prototypes) # get the length of the output

    time_total = np.arange(1, lenght_out+1, 1) 
    time_train = np.arange(1, n_per_prototype_train * n_main_classes+1, 1)
    time_test = np.arange(1, (n_per_prototype_test + off_time * n_per_prototype_test ) * n_main_classes+1, 1)
    
    time_train_total = time_train

    batch_idx = np.arange(0, num_prototypes, n_per_prototype)
    batch_train = np.concatenate(train_test_data_finnegan[0][batch_idx])[np.random.choice(np.arange(0, n_per_prototype_train*n_main_classes), n_per_prototype_train*n_main_classes, replace=False)]
    batch_test =  np.concatenate(train_test_data_finnegan[1][batch_idx])
    Ids_train = clamped_input_transform(batch_train, min_p = 1e-4, max_p = .95)
    Ids_test = create_test_Id(batch_test, off_time=off_time)
    ids_test_final = Ids_test
    Ids = np.row_stack((np.zeros(num_vis), Ids_train, Ids_test))
    for i in np.arange(n_per_prototype-1)+1:
        batch_train = np.concatenate(train_test_data_finnegan[0][batch_idx+i])[np.random.choice(np.arange(0, n_per_prototype_train*n_main_classes), n_per_prototype_train*n_main_classes, replace=False)]
        batch_test =  np.concatenate(train_test_data_finnegan[1][batch_idx+i])
        Ids_train = clamped_input_transform(batch_train, min_p = 1e-4, max_p = .95)
        Ids_test = create_test_Id(batch_test, off_time=off_time)
        ids_test_final = np.row_stack((ids_test_final, Ids_test))
        Ids = np.row_stack((Ids, Ids_train, Ids_test))

        time_train_total = np.append(time_train_total, time_train + time_train_total[-1] + time_test[-1])

    Ids = np.row_stack((Ids, ids_test_final))
    time_test_total = np.setdiff1d(time_total, time_train_total)

    time_test_on = time_test_total[::off_time+1]
    time_test_off = np.setdiff1d(time_test_total, time_test_on)

    return Ids, time_test_on, time_test_off

def orthogonalization_degree(x, y):
    from scipy.stats import pearsonr
    return((1-np.round(pearsonr(x, y)[0], 15))/2)

def pattern_distance(ortho, av_act):
    return(ortho/av_act)

def pattern_separation_efficacy(input_1, input_2, output_1, output_2):
    orthogonalization_input = orthogonalization_degree(input_1, input_2)
    orthogonalization_output = orthogonalization_degree(output_1, output_2)
    av_activ_input = average_activation_degree(input_1, input_2)
    av_activ_output = average_activation_degree(output_1, output_2)
    pattern_distance_input = pattern_distance(orthogonalization_input, av_activ_input)
    pattern_distance_output = pattern_distance(orthogonalization_output, av_activ_output)
    return(pattern_distance_output/pattern_distance_input)

def load_output(unique = "output", date = "", path = "output/"):
    import pickle
    name = unique+date
    with open(path+name+".pkl", 'rb') as handle:
        output = pickle.load(handle)
    return output

def binarize(x, threshold):
    return np.where(x > threshold, 1, 0)

def pattern_separation_efficacy_model(monitor_vis, monitor_hid, timpoint_dict, n_seed_patterns, n_prototype_per_seed, after_split_n_per_prototype_test, time_test_on, selection = "group", to_binary = False, 
                                      convert_to_hz = False, threshold = 1, report = False):
    test_patterns_while_training = after_split_n_per_prototype_test * n_prototype_per_seed * n_seed_patterns # get the number of test patterns while training
    final_test_patterns = time_test_on[test_patterns_while_training:] # get test patterns after training was done (its the same number as test patterns while training)
    if selection == "group": 
        out = np.zeros((n_prototype_per_seed, n_seed_patterns, n_seed_patterns)) # initialize output 
        for i in range(n_prototype_per_seed):
            test_patterns_group = final_test_patterns[i:i+n_seed_patterns] 
            for j in range(n_seed_patterns):
                for k in range(n_seed_patterns):
                    t_1 = test_patterns_group[j]
                    t_2 = test_patterns_group[k]
                    timepoint_s_1 = timpoint_dict["T"+str(t_1)+"_s"]
                    timepoint_e_1 = timpoint_dict["T"+str(t_1)+"_e"]
                    timepoint_s_2 = timpoint_dict["T"+str(t_2)+"_s"]
                    timepoint_e_2 = timpoint_dict["T"+str(t_2)+"_e"]

                    sv1 = spike_histogram(monitor_vis, timepoint_s_1, timepoint_e_1, s_per_s=convert_to_hz).T[1]
                    sh1 = spike_histogram(monitor_hid, timepoint_s_1, timepoint_e_1, s_per_s=convert_to_hz).T[1]
                    sv2 = spike_histogram(monitor_vis, timepoint_s_2, timepoint_e_2, s_per_s=convert_to_hz).T[1]
                    sh2 = spike_histogram(monitor_hid, timepoint_s_2, timepoint_e_2, s_per_s=convert_to_hz).T[1]

                    if report:
                        print("sv1", sv1)

                    if to_binary:
                        sv1 = binarize(sv1, threshold = threshold)
                        sh1 = binarize(sh1, threshold = threshold)
                        sv2 = binarize(sv2, threshold = threshold)
                        sh2 = binarize(sh2, threshold = threshold)
                        if report:
                            print("Binarized to: ", sv1)

                    if j != k:
                        out[i, j, k] = pattern_separation_efficacy(sv1, sv2, sh1, sh2)
    if selection == "prototype":
        out = np.zeros((n_seed_patterns, n_prototype_per_seed, n_prototype_per_seed))
        for i in range(n_seed_patterns):
            test_patterns_group = final_test_patterns[i::n_seed_patterns]
            for j in range(n_prototype_per_seed):
                for k in range(n_prototype_per_seed):
                    t_1 = test_patterns_group[j]
                    t_2 = test_patterns_group[k]
                    timepoint_s_1 = timpoint_dict["T"+str(t_1)+"_s"]
                    timepoint_e_1 = timpoint_dict["T"+str(t_1)+"_e"]
                    timepoint_s_2 = timpoint_dict["T"+str(t_2)+"_s"]
                    timepoint_e_2 = timpoint_dict["T"+str(t_2)+"_e"]

                    sv1 = spike_histogram(monitor_vis, timepoint_s_1, timepoint_e_1, s_per_s=convert_to_hz).T[1]
                    sh1 = spike_histogram(monitor_hid, timepoint_s_1, timepoint_e_1, s_per_s=convert_to_hz).T[1]
                    sv2 = spike_histogram(monitor_vis, timepoint_s_2, timepoint_e_2, s_per_s=convert_to_hz).T[1]
                    sh2 = spike_histogram(monitor_hid, timepoint_s_2, timepoint_e_2, s_per_s=convert_to_hz).T[1]

                    if report:
                        print("sv1", sv1)

                    if to_binary:
                        sv1 = binarize(sv1, threshold = threshold)
                        sh1 = binarize(sh1, threshold = threshold)
                        sv2 = binarize(sv2, threshold = threshold)
                        sh2 = binarize(sh2, threshold = threshold)
                        if report:
                            print("Binarized to: ", sv1)

                    if j != k:
                        out[i, j, k] = pattern_separation_efficacy(sv1, sv2, sh1, sh2)
    return out

def create_connection_matrix(N_input, N_hidden, probabilities, pmin = 0, pmax = 1):
    '''Creates a binary connection matrix with the number of connetions between the
      hidden and visible neurons beeing dependent on the given probabilities for the hidden neurons.'''
    if np.min(probabilities) != np.max(probabilities):
        probabilities = (probabilities - np.min(probabilities))/(np.max(probabilities) - np.min(probabilities))*(pmax-pmin) + pmin
    W = np.zeros((N_input, N_hidden))
    for i in range(N_hidden):
        for j in range(N_input):
            if np.random.random() < probabilities[i]:
                W[j,i] = 1
    return W

def update_connection_matrix(connections, probabilities, pmin = 0, pmax = 1):
    '''Updates a binary connection matrix with the number of connetions between the
      hidden and visible neurons beeing dependent on the given probabilities for the hidden neurons.'''
    # scale probabilities to be between pmin and pmax
    if np.min(probabilities) != np.max(probabilities):
        probabilities = (probabilities - np.min(probabilities))/(np.max(probabilities) - np.min(probabilities))*(pmax-pmin) + pmin
    else:
        probabilities = np.repeat(pmax, len(probabilities))
    for i in range(connections.shape[1]):
        n_active = np.sum(connections[:,i])
        missing = int(np.round(probabilities[i]*connections.shape[0])) - n_active
        if missing > 0:
            idx = np.random.choice(np.where(connections[:,i]==0)[0], size=int(missing), replace=False)
            connections[idx,i] = 1
        elif missing < 0:
            idx = np.random.choice(np.where(connections[:,i]==1)[0], size=int(-missing), replace=False)
            connections[idx,i] = 0
    return connections


def get_hist_vis_hidden(spike_monitor_visible, spike_monitor_hidden, times, time_points_dict, normalize = True, binarize = False, threshold = 0.5): 
    vis = []
    hid = []
    for i in times:
        t_start = time_points_dict["T"+ str(i)+"_s"]
        t_stop = time_points_dict["T"+ str(i)+"_e"]
        visible = spike_histogram(spike_monitor_visible, t_start=t_start, t_stop=t_stop).T[1]
        hiddden = spike_histogram(spike_monitor_hidden, t_start=t_start, t_stop=t_stop).T[1]
        if normalize:
            visible = normalizer(visible)
            hiddden = normalizer(hiddden)
        if binarize:
            visible = np.where(visible > threshold, 1, 0)
            hiddden = np.where(hiddden > threshold, 1, 0)
        vis.append(visible)
        hid.append(hiddden)
    return vis, hid


def plot_input_output_curves(outputs, model_identifyer, alpha = 0.5, threshold = 0.5, normalize = True, binarize = False, order_of_model = 3, off_time = 1, 
                             plot_3rd_order = False, plot_error_bars = True, split = (0.5,1), ylimit = (0, 100), xlimit = (0, 20), 
                             go_through_origin = False, 
                             colors = [[0.941, 0.62 , 0.137], [0.216, 0.639, 0.82], [0.875, 0.22 , 0.09], [0.322, 0.714, 0.345], [0.788, 0.373, 0.773]]):
    ''' Sorry for this horrible mess of a function. It is used to plot the input-output curves of the network.'''
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    if isinstance(threshold, float):
        threshold = np.repeat(threshold, len(outputs))

    counter = 0
    space = -0.5
    legend_elements = []

    for output in outputs:
        Mv_loaded = output["Mv"]
        Mh_loaded = output["Mh"]
        time_test_on = output["time_test_on"]
        time_points_dict = output["time_points_dict"]

        time_test_off_loaded = output["time_test_off"]
        t_on = np.setdiff1d(np.arange(1, max(time_test_off_loaded)), time_test_off_loaded)
        time_points = t_on[int(len(t_on)*split[0]):int(len(t_on)*split[1])]

        originals, recovered = get_hist_vis_hidden(Mv_loaded, Mh_loaded, time_points, time_points_dict, normalize = normalize,
                                                    binarize = binarize, threshold = threshold[counter])
        distances_in = np.zeros((len(originals), len(originals)))
        for i in range(len(originals)):
            for j in range(len(originals)):
                distances_in[i,j] = calculate_hamming_distance(originals[i], originals[j])

        distances_out = np.zeros((len(recovered), len(recovered)))
        for i in range(len(recovered)):
            for j in range(len(recovered)):
                distances_out[i,j] = calculate_hamming_distance(recovered[i], recovered[j])
        unique_dist_in = np.unique(distances_in)
        matched_dist_out_mean = []
        matched_dist_out_std = []
        all_in = []
        all_out = []
        for i in unique_dist_in:
            matched_dist_out_mean.append(mean(distances_out[np.where(distances_in == i)]))
            matched_dist_out_std.append(std(distances_out[np.where(distances_in == i)]))
            all_in.append(distances_in[np.where(distances_in == i)])
            all_out.append(distances_out[np.where(distances_in == i)])

        all_in = np.concatenate(all_in)
        all_out = np.concatenate(all_out)


        model_2 = np.poly1d(np.polyfit(all_in, all_out, 1))
        polyline = np.linspace(-2, max(unique_dist_in)+1, 100)
        col = colors[counter][:3]
        col.append(alpha)
        if plot_error_bars:
            plt.errorbar(unique_dist_in+space, matched_dist_out_mean, matched_dist_out_std, linestyle='None', marker='.', linewidth=1,
                        color = col)
        else:
            plt.plot(unique_dist_in+space, matched_dist_out_mean, linestyle='None', marker='.', markersize=5,
                        color = col)

        if plot_3rd_order:
            from scipy.optimize import curve_fit
            if go_through_origin:
                def fit_func(x, a, b, c):
                    # Curve fitting function
                    return a * x**3 + b * x**2 + c * x  # d=0 is implied

                # Curve fitting
                params = curve_fit(fit_func, all_in, all_out)
                [a, b, c] = params[0]
                x_fit = np.linspace(all_in[0], all_in[-1], 100)
                y_fit = a * x_fit**3 + b * x_fit**2 + c * x_fit 
            else: 
                def fit_func(x, a, b, c, d):
                    # Curve fitting function
                    return a * x**3 + b * x**2 + c * x + d # d=0 is implied

                """ plt.plot(polyline, model_1(polyline), color = mcolors.BASE_COLORS[list(mcolors.BASE_COLORS)[counter]], 
                        linestyle='-', linewidth=1) """
                # Curve fitting
                params = curve_fit(fit_func, all_in, all_out)
                [a, b, c, d] = params[0]
                x_fit = np.linspace(all_in[0], all_in[-1], 100)
                y_fit = a * x_fit**3 + b * x_fit**2 + c * x_fit + d
            plt.plot(x_fit, y_fit, label="_lalala", color = col[:3])  # Fitted curve
        else: 
            plt.plot(polyline, model_2(polyline), color = col[:3], 
                linestyle='--', linewidth=0.5)  
        
        plt.xlim(xlimit)
        plt.ylim(ylimit)
        plt.xlabel("Hamming distance between input patterns")
        plt.ylabel("Hamming distance between hidden patterns")
        legend_elements.append(Line2D([0], [0], marker='o', color="w", label=model_identifyer[counter],  markerfacecolor=col[:3], markersize=5))
        counter += 1
        space += 0.5
    #plt.legend(model_identifyer[:len(outputs)])
    plt.legend(handles=legend_elements)
    plt.plot(np.linspace(-2, max(ylimit)+1, 3), np.linspace(-2, max(ylimit)+1, 3), color='gray', 
            linestyle='--', linewidth=1)

    plt.show()

