import numpy as np

def exp_prob_beta_gamma(dt, beta, g_leak, gamma, t_ref):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func

def bound_data(data, min_p = 0.0001, max_p = .95, binary = False):
    if not binary:
        max_p_ = max_p
        min_p_ = min_p
    else:
        max_p_ = 0.5
        min_p_ = 0.5
    data[data >= max_p_] = max_p
    data[data < min_p_] = min_p
    
def select_equal_n_labels(n, data, labels, classes = None, seed=None):
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
