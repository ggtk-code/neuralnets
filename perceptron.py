import numpy as np

class Perceptron(object):
    """Perceptron binary classifier
    A perceptron is a basic neural network classifier. Here are the definitions of
    various components:
    
    Input: is a vector of reals, represented by X. During training, there are many 
    different such training samples, represented by xi (i in superscript)
    
    Weights: Each attribute / fature in the vector is given a weight wj. w0 represents
    the constant. 
    
    Net Input: This is the weighted sum of the inputs: w0 + w1x_1 + .. + wjx_j + wmx_m

    Activation Function: This takes the net input and transforms before final classification.
    
    Prediction: Takes the output of the activation funtion and produces a binary output {-1, +1}

    Fit: Takes a bunch of training samples X and the expected outputs y, and adjusts the
    weight vector. In a perceptron, each sample is used individually to update weights. 
    That is: w -sample1 --> w' --sample2--> w'' --sample3--> w''' etc

    """

    # Meta params are: number of iterations, learning rate, 
    def __init__(self, num_iter, learning_rate):
        self.n_iter_ = num_iter
        self.eta_ = learning_rate
        # Shape of w will depend on the training data
        # Model is w0 + w1x1 + w2x2 + ... + wnxn and threshold = 0
        self.w_ = None
        # Num of incorrectly predicted samples
        self.errors_ = []

    """ Train the perception.
    X is a numpy array with shape num_samples X num_features
    y is the expected output / class labels. Typically, each element is -1 or +1
    """
    def Fit(self, X, y):
        nsamples = X.shape[0]
        nfeatures = X.shape[1]
        print("X has shape " + str(np.shape(X)))
        print("y has shape " + str(np.shape(y)))
        # Randomize 1 + nfeatures -- the extra 1 is for the constant so we can assume threshold
        # is 0
        self.w_ = np.random.RandomState().normal(loc = 0.0, scale = 0.01, size = 1 + nfeatures)
        print("weights initialized -- shape is " + str(np.shape(self.w_)))
        self.PrintModel()
        self.Accuracy(X, y)
        self.errors_ = []
        
        # Iterate several times
        for _ in range(self.n_iter_):
            errors = 0
            for xi, target in zip(X, y):
                # xi is the ith sample, target is the target y
                # weights will be updated proportionally to both the learning rate
                # and how far off the target is from prediction
                # Note that target and self.Predict() are both binary (-1 or +1)
                # so the possible values of update_factor are 0, +2*eta, -2*eta
                update_factor = self.eta_ * (target - self.Predict(xi))
                if update_factor != 0.0:
                    errors += 1
                #print("update_factor for " + str(xi) + " = " + str(update_factor))
                self.w_[1:] += update_factor * xi
                self.w_[0] += update_factor
            self.errors_.append(errors)
        print("Training complete. Classification errors (out of " + str(nsamples) + "):")
        print(str(self.errors_))

    
    def NetInput(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def Activation(self, X):
        return self.NetInput(X)
    
    def Predict(self, X):
        val = self.Activation(X)
        if val >= 0.0:
            return 1
        else:
            return -1

    def PrintModel(self):
        print("Model is: " + str(self.w_))
        
            

    # what is the accuracy of the current model
    # wrt input X and class labels y
    def Accuracy(self, X, y):
        errors = 0
        for xi, target in zip(X, y):
            if target != self.Predict(xi):
                errors += 1
        nsamples = X.shape[0]
        ncorrect = nsamples - errors
        print("accuracy = " + str(ncorrect) + " / " + str(nsamples))
        return ncorrect

def test():
    # Generate samples for x1 + x2 + x3 + 1.0 >= 0.0
    num_samples = 1000
    X = []
    y = []
    npos_samples = 0
    nneg_samples = 0
    for _ in range(num_samples):
        sample = np.random.RandomState().normal(loc = -1.0, scale = 1.0, size = 3)
        #print("sample=" + str(sample))
        val = np.dot([1.0, 1.0, 1.0], sample) + 1.0
        X.append(sample)
        if val >= 0:
            y.append(1)
            npos_samples += 1
        else:
            y.append(-1)
            nneg_samples += 1

    nX = np.array(X)
    ny = np.array(y)
    #print("X = ")
    #print(nX)
    #print("y = ")
    #print(ny)
    print("positive samples: " + str(npos_samples) + " negative samples: " + str(nneg_samples))
    print("=======================")

    # Initialize with hyperparams
    p = Perceptron(10, 0.1)
    p.Fit(nX, ny)
    p.PrintModel()
    p.Accuracy(nX, ny)

    # Adding a random comment
        
def main():
    # Run test
    test()


        
if __name__ == "__main__":
    # execute only if run as a script
    main()
