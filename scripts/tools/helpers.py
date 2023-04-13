import numpy as np
from tools.parameters import *

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
    '''Selects n samples from each class.'''
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

def load_MNIST(n_samples, min_p = 0.0001, max_p = .95, binary = False, seed=None, datafile = path_to_data, num_classes = range(10)):
    import gzip, pickle
    mat = pickle.load(gzip.open(datafile, 'r'), encoding='latin1')

    train_iv = mat['train']
    train_iv_l = mat['train_label']
    test_iv = mat['test']
    test_iv_l = mat['test_label']
    
    bound_data(train_iv, min_p, max_p, binary)
    bound_data(test_iv, min_p, max_p, binary)
    
    iv_seq, iv_l_seq = select_equal_n_labels(n_samples, train_iv, train_iv_l, seed = seed, classes=num_classes)
    
    return iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l

def load_mnist_data(min_p = 1e-4, max_p=.95, binary=False, seed=None, n_classes = range(10)):
    #------------------------------------------ Create Input Vector
    mnist_data = load_MNIST(n_samples,
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

def create_Id(data = True, c_min_p = 1e-4, c_max_p = .95, seed = None):
    '''Creates the input vector sequence with the clamped input. If data is True, then the MNIST data is loaded. 
       If data is a tuple, then the data is used. If data is False, then the input vector sequence is all zeros.'''
    if hasattr(data, '__len__'):
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
        Idp = create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = c_min_p, max_p = c_max_p)
        Id = (Idp /beta)
    elif data == True:
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = load_mnist_data(seed = seed)
        Idp = create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = c_min_p, max_p = c_max_p)
        Id = (Idp /beta)
    else:
        Id = np.zeros([n_samples,N_v+N_c])  
    return Id

def classification_free_energy(Wvh, Wch, b_h, b_c, test_data, test_labels, n_c_unit, n_classes = 10):
    '''Classifies the label based on the free energy of the model given the test data.'''
    numcases = len(test_labels);
    F = np.zeros([int(numcases), int(n_classes)]);
    for i in range(n_classes):
        X= np.zeros([int(numcases), int(n_c_unit)*int(n_classes)]);
        X[:, int(n_c_unit*i):int(n_c_unit*(i+1))] = 1;
        F[:,i] = np.tile(b_c[i],numcases)*X[:,i]+\
                 np.sum(np.log(np.exp(np.dot(test_data, Wvh)+np.dot(X,Wch)+np.tile(b_h,numcases).reshape(numcases,-1))+1), axis=1);
    prediction= np.argmax(F, axis=1);
    accuracy = sum(prediction==test_labels)/numcases # changed from: 1-float(sum(prediction!=test_labels))/numcases
    assert 1>=accuracy>=.1/n_classes
    return accuracy, prediction!=test_labels

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

def create_rbm_parameters(wmean=0, b_vmean=0, b_hmean=0):
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

