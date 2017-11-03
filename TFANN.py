'''
TFANN.py
This file contains classes implementing ANN using numpy 
and tensorflow. Presently, multi-layer perceptron (MLP) 
and convolutional neural networks (CNN) are supported.
'''
import tensorflow as tf
import numpy as np

def _Accuracy(Y, YH):
    '''
    Return the classification accuracy
    given a vector of target labels and
    predicted labels
    y: The target labels
    yHat: The predicted labels
    return: The percentage correct
    '''
    return np.mean(Y == YH)
    
def _CreateCNN(AF, PM, WS, X):
    '''
    Sets up the graph for a convolutional neural network from a list of specifications 
    of the form:
    [('C', [5, 5, 3, 64], [1, 1, 1, 1]), ('P', [1, 3, 3, 1], [1, 2, 2, 1]),('F', 10),]
    Where 'C' denotes a covolution layer, 'P' denotes a pooling layer, and 'F' denotes
    a fully-connected layer.
    AF:     The activation function
    pm:     The padding method
    ws:     The specifications describe above
    X:      Input tensor
    '''
    W, B, O = [], [], [X]
    for i, WSi in enumerate(WS):
        if WSi[0] == 'C':           #2-D Convolutional layer
            W.append(tf.Variable(tf.truncated_normal(WSi[1], stddev = 5e-2)))
            B.append(tf.Variable(tf.constant(0.0, shape = [WSi[1][-1]])))
            X = tf.nn.conv2d(X, W[-1], WSi[2], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
            if i + 1 != len(WS):    #Last layer shouldn't apply activation function
                X = AF(X)
        if WSi[0] == 'CT':          #2-D Convolutional Transpose layer
            print(WSi)
            W.append(tf.Variable(tf.truncated_normal(WSi[1], stddev = 5e-2)))
            B.append(tf.Variable(tf.constant(0.0, shape = [WSi[1][-2]])))
            X = tf.nn.conv2d_transpose(X, W[-1], WSi[2], WSi[3], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
            if i + 1 != len(WS):    #Last layer shouldn't apply activation function
                X = AF(X)
            print(X.shape)
        if WSi[0] == 'C1d':         #1-D Convolutional Layer
            W.append(tf.Variable(tf.truncated_normal(WSi[1], stddev = 5e-2)))
            B.append(tf.Variable(tf.constant(0.0, shape = [WSi[1][-1]])))
            X = tf.nn.conv1d(X, W[-1], WSi[2], padding = PM)
            X = tf.nn.bias_add(X, B[-1])
            if i + 1 != len(WS):    #Last layer shouldn't apply activation function
                X = AF(X)
        elif WSi[0] == 'P':         #Pooling layer
            X = tf.nn.max_pool(X, ksize = WSi[1], strides = WSi[2], padding = PM)
            X = tf.nn.lrn(X, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        elif WSi[0] == 'F':         #Fully-connected layer
            if tf.rank(X) != 2:
                X = tf.contrib.layers.flatten(X)
            lls = X.get_shape()[1].value
            W.append(tf.Variable(tf.truncated_normal([lls, WSi[1]], stddev = 0.04)))
            B.append(tf.Variable(tf.constant(0.1, shape = [WSi[1]])))
            X = tf.matmul(X, W[-1]) + B[-1]
            if i + 1 != len(WS):    #Last layer shouldn't apply activation function
                X = AF(X)
        elif WSi[0] == 'R':         #Reshape layer
            X = tf.reshape(X, WSi[1])
        O.append(X)
    return O, W, B

def _CreateMLP(_X, _W, _B, _AF):
    '''
    Create the MLP variables for TF graph
    _X: The input matrix
    _W: The weight matrices
    _B: The bias vectors
    _AF: The activation function
    '''
    n, O = len(_W), [_X]
    for i in range(n - 1):
        O.append(_AF(tf.matmul(O[-1], _W[i]) + _B[i]))
    return O + [tf.matmul(O[-1], _W[n - 1]) + _B[n - 1]]

def _CreateRFSMLP(_X, _W, _B, _AF):
    '''
    Create the RFSMLP variables for TF graph
    _X: The input matrix
    _W: The weight matrices
    _B: The bias vectors
    _AF: The activation function
    '''
    n, X = len(_W), [_X]
    for i in range(n - 1):
        X.append(_AF(tf.matmul(X[-1], _W[i]) + _B[i]))
    return tf.stack(X + [tf.matmul(X[-1], _W[n - 1]) + _B[n - 1]], axis = 1)

def _CreateL2Reg(_W, _B):
    '''
    Add L2 regularizers for the weight and bias matrices
    _W: The weight matrices
    _B: The bias matrices
    return: tensorflow variable representing l2 regularization cost
    '''
    n = len(_W)
    regularizers = tf.nn.l2_loss(_W[0]) + tf.nn.l2_loss(_B[0])
    for i in range(1, n):
        regularizers += tf.nn.l2_loss(_W[i]) + tf.nn.l2_loss(_B[i])
    return regularizers

def _CreateVars(layers):
    '''
    Create weight and bias vectors for an MLP
    layers: The number of neurons in each layer (including input and output)
    return: A tuple of lists of the weight and bias matrices respectively
    '''
    weight = []
    bias = []
    n = len(layers)
    for i in range(n - 1):
        lyrstd = np.sqrt(1.0 / layers[i])           #Fan-in for layer; used as standard dev
        curW = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], stddev = lyrstd))
        weight.append(curW)
        curB = tf.Variable(tf.random_normal([layers[i + 1]], stddev = lyrstd))
        bias.append(curB)

    return (weight, bias)

def _GetActvFn(name):
    '''
    Helper function for selecting an activation function
    name: The name of the activation function
    return: A handle for the tensorflow activation function
    '''
    if name == 'tanh':
        return tf.tanh
    elif name == 'sig':
        return tf.sigmoid
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'relu6':
        return tf.nn.relu6
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'softsign':
        return tf.nn.softsign
    return None
    
def _GetBatchRange(bs, mIter):
    '''
    Gives a range from which to choose the batch size
    '''
    try:
        return np.linspace(*bs[0:2], num = mIter).round().astype(np.int)
    except TypeError:
        pass
    return np.repeat(bs, mIter)
    
def _GetLossFn(name):
    '''
    Helper function for selecting loss function
    name:   The name of the loss function
    return:     A handle for a loss function LF(YH, Y)
    '''
    if name == 'l2':
        return lambda YH, Y : tf.squared_difference(Y, YH)
    elif name == 'l1':
        return lambda YH, Y : tf.losses.absolute_difference(Y, YH)
    elif name == 'smce':
        return lambda YH, Y : tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = YH)
    elif name == 'sgce':
        return lambda YH, Y : tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = YH)
    elif name == 'cos':
        return lambda YH, Y : tf.losses.cosine_distance(Y, YH)
    elif name == 'log':
        return lambda YH, Y : tf.losses.log_loss(Y, YH)
    elif name == 'hinge':
        return lambda YH, Y : tf.losses.hinge_loss(Y, YH)
    return None

def _GetOptimizer(name, lr):
    '''
    Helper function for getting a tensorflow optimizer
    name:   The name of the optimizer to use
    lr:   The learning rate if applicable
    return;  A the tensorflow optimization object
    '''
    if name == 'adam':
        return tf.train.AdamOptimizer(learning_rate = lr)
    elif name == 'grad':
        return tf.train.GradientDescentOptimizer(learning_rate = lr)
    elif name == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate = lr)
    elif name == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate = lr)
    return None

class ANN:
    sess = None     #Class variable shared among all instances
    sessc = 0       #Number of active sessions
    '''
    TensorFlow Artificial Neural Network (Base Class)
    '''

    #Clean-up resources
    def __del__(self):
        ANN.sessc -= 1          #Decrement session counter
        if ANN.sessc == 0:      #Close session only if not in use
            self.GetSes().close()
            ANN.sess = None

    def __init__(self, actvFn = 'relu', batchSize = None, learnRate = 1e-4, loss = 'l2', maxIter = 1024,
                 name = 'tfann', optmzrn = 'adam', reg = None, tol = 1e-2, verbose = False, X = None, Y = None):
        '''
        Common arguments for all artificial neural network regression models
        actvFn:     The activation function to use: 'tanh', 'sig', or 'relu'
        batchSize:  Size of training batches to use (use all if None)
        learnRate:  The learning rate parameter for the optimizer
        loss:       The name of the loss function (l2, l1, smce, sgme, cos, log, hinge) 
        name:       The name of the model for variable scope, saving, and restoring
        maxIter:    Maximum number of training iterations
        optmzrn:    The optimizer method to use ('adam', 'grad', 'adagrad', or 'ftrl')
        reg:        Weight of the regularization term (None for no regularization)
        tol:        Training ends if error falls below this tolerance
        verbose:    Print training information
        X:          Tensor that is used as input to model; if None a new
                    placeholder is created and used. If specified a feed_dict
                    must be passed to train, predict, and score calls!
        '''
        self.AF = _GetActvFn(actvFn)                #Activation function to use
        self.LF = _GetLossFn(loss)                  #Handle to loss function to use
        self.batSz = batchSize                      #Batch size
        self.init = None                            #Initializer operation
        self.lr = learnRate                         #Learning rate
        self.mIter = maxIter                        #Maximum number of iterations
        self.name = name                            #Name of model for variable scope
        self.optmzrn = optmzrn                      #Optimizer method name
        self.reg = reg                              #Regularization strength
        self.stopIter = False                       #Flag for stopping training early
        self.tol = tol                              #Error tolerance
        self.vrbse = verbose                        #Verbose output
        self.X = X                                  #TF Variable used as input to model
        self.Y = Y                                  #TF Variable used as target for model
        #Data members to be populated in a subclass
        self.loss = None
        self.optmzr = None
        self.O = None
        self.TFVar = None
        with tf.variable_scope(self.name):
            self.TFVar = set(tf.global_variables())                 #All variables before model
            self.CreateModel()                                      #Populate TensorFlow graph
            self.TFVar = set(tf.global_variables()) - self.TFVar    #Only variables in this model
            self.Initialize()                                       #Start TF session and initialize vars 
            
    def CreateModel(self):
        '''
        Creates the acutal model (implemented in subclasses)
        '''
        pass
        
    def fit(self, A, Y, FD = None):
        '''
        Fit the ANN to the data
        A: numpy matrix where each row is a sample
        Y: numpy matrix of target values
        '''
        m = len(A)
        FD = {self.X:A, self.Y:Y} if FD is None else FD     #Feed dictionary
        #Loop up to mIter times gradually scaling up the batch size (if range is provided)
        for i, BSi in enumerate(_GetBatchRange(self.batSz, self.mIter)):
            if BSi is None:                          #Compute loss and optimize simultaneously for all samples
                err, _ = self.GetSes().run([self.loss, self.optmzr], feed_dict = FD)
            else:                                           #Train m samples using random batches of size self.bs
                err = 0.0
                for j in range(0, m, BSi):                  #Compute loss and optimize simultaneously for batch
                    bi = np.random.choice(m, BSi, False)    #Randomly chosen batch indices
                    BFD = {k:v[bi] for k, v in FD.items()}  #Feed dictionary for this batch
                    l, _ = self.GetSes().run([self.loss, self.optmzr], feed_dict = BFD)
                    err += l                                #Accumulate loss over all batches
                err /= len(range(0, m, BSi))                #Average over all batches
            if self.vrbse:
                print("Iter {:5d}\t{:16.8f} (Batch Size: {:5d})".format(i + 1, err, -1 if BSi is None else BSi))
            if err < self.tol or self.stopIter:
                break   #Stop if tolerance was reached or flag was set
                
    def GetBias(self, i):
        '''
        Warning! Does not work with restored models currently.
        Gets the i-th bias vector from the model
        '''
        return self.B[i].eval(session = self.GetSes())
                
    def GetSes(self):
        if ANN.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True      #Only use GPU memory as needed
            ANN.sess = tf.Session(config = config)      #instead of allocating all up-front
        return ANN.sess
                
    def GetWeightMatrix(self, i):
        '''
        Warning! Does not work with restored models currently.
        Gets the i-th weight matrix from the model.
        '''
        return self.W[i].eval(session = self.GetSes())

    def Initialize(self):
        '''
        Initialize variables and start the TensorFlow session
        '''
        ANN.sessc += 1                              #Increment session counter
        sess = self.GetSes()
        self.init = tf.variables_initializer(self.TFVar)
        sess.run(self.init)                         #Initialize all variables for this model
        
    def predict(self, A, FD = None):
        '''
        Predict the output given the input (only run after calling fit)
        A: The input values for which to predict outputs
        return: The predicted output values (one row per input sample)
        '''
        FD = {self.X:A} if FD is None else FD
        return self.GetSes().run(self.O[-1], feed_dict = FD)
        
    def PredictFull(self, A, FD = None):
        '''
        Predict the output given the input (only run after calling fit)
        returning all intermediate results
        A: The input values for which to predict outputs
        return: The predicted output values with intermediate results
        (one row per input sample)
        '''
        FD = {self.X:A} if FD is None else FD
        return self.GetSes().run(self.O, feed_dict = FD)

    def Reinitialize(self):
        '''
        Reinitialize variables and start the TensorFlow session
        '''
        sess = self.GetSes()
        sess.run(self.init) #Re-Initialize all variables for this model
        
    def Reset():
        '''
        Performs a hard reset of the tensorflow graph and session
        '''
        ANN.sessc = 0
        if ANN.sess is not None:
            ANN.sess.close()
        ANN.sess = None
        tf.reset_default_graph()
        
    def RestoreModel(self, p, n):
        '''
        Restores a model that was previously created with SaveModel
        p:      Path to model containing files like "path/"
        '''
        try:
            saver = tf.train.Saver()
            saver.restore(self.GetSes(), p + n)
        except Exception as e:
            print('Error restoring: ' + p + n)
            return False
        return True

    def SaveModel(self, p):
        '''
        Save model to file
        p:  Path to save file like "path/"
        '''
        saver = tf.train.Saver()
        saver.save(self.GetSes(), p)

    def SetMaxIter(self, mi):
        '''
        Update the maximum number of iterations to run for fit
        '''
        self.mIter = mi
        
    def SetStopIter(self, si):
        '''
        Set the flag for stopping iteration early
        '''
        self.stopIter = si
        
    def Tx(self, Y):
        '''
        Map from {0, 1} -> {-0.9, .9} (for tanh activation)
        '''
        return (2 * Y - 1) * 0.9
    
    def TInvX(self, Y):
        '''
        Map from {-0.9, .9} -> {0, 1}
        '''
        return (Y / 0.9 + 1) / 2.0
    
    def YHatF(self, Y):
        '''
        Convert from prediction (YH) to original data type (integer 0 or 1)
        '''
        return Y.clip(0.0, 1.0).round().astype(np.int)

class ANNR(ANN):
    '''
    Artificial Neural Network for Regression (Base Class)
    '''
   
    def score(self, A, Y, FD = None):
        '''
        Predicts the loss function for given input and target matrices
        A: The input data matrix
        Y: The target matrix
        return: The loss
        '''
        FD = {self.X:A, self.Y:Y} if FD is None else FD
        return self.GetSes().run(self.loss, feed_dict = FD)

class MLPR(ANNR):
    '''
    Multi-Layer Perceptron for Regression
    '''
    
    def __init__(self, layers, actvFn = 'tanh', batchSize = None, learnRate = 1e-3, loss = 'l2', maxIter = 1024, 
                 name = 'tfann', optmzrn = 'adam',  reg = 1e-3, tol = 1e-2, verbose = False, X = None, Y = None):
        '''
        layers: A list of layer sizes
        '''
        self.layers = layers
        super().__init__(actvFn, batchSize, learnRate, loss, maxIter, name, optmzrn, reg, tol, verbose, X, Y)

    def CreateModel(self):
        if self.X is None:  #If no input variable was specified make a new placeholder
            self.X = tf.placeholder("float", [None, self.layers[0]])    #Input data matrix
        self.W, self.B = _CreateVars(self.layers)                       #Setup the weight and bias variables
        self.O = _CreateMLP(self.X, self.W, self.B, self.AF)   #Create the tensorflow MLP model; YH is graph output
        if self.Y is None:
            self.Y = tf.placeholder("float", self.O[-1].shape)     #Target value matrix
        self.loss = tf.reduce_mean(self.LF(self.O[-1], self.Y))                
        if self.reg is not None:                                            #Regularization can prevent over-fitting
            self.loss += _CreateL2Reg(self.W, self.B) * self.reg
        self.optmzr = _GetOptimizer(self.optmzrn, self.lr).minimize(self.loss)  #Get optimizer method to minimize the loss function    