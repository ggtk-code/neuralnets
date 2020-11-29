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

    # Meta params are: number of iterations, learning rate
    # If std_features = True, then the data is first transformed such that each
    # feature has 0 mean and 1 std deviation
    def __init__(self, num_iter, learning_rate, std_features = False):
        self.n_iter_ = num_iter
        self.eta_ = learning_rate
        # standardize refers to making f
        self.std_features_ = std_features
        # Shape of w will depend on the training data
        # Model is w0 + w1x1 + w2x2 + ... + wnxn and threshold = 0
        self.w_ = None
        # Standardization params
        self.mean_ = None
        self.stddev_ = None
        # Num of incorrectly predicted samples
        self.errors_ = []

    """ Train the perception.
    X is a numpy array with shape num_samples X num_features
    y is the expected output / class labels. Typically, each element is -1 or +1
    """
    def Fit(self, X, y):
        # cleanup X and y if needed
        if self.std_features_:
            X = self.Standardize(X)
        self.FitInternal(X, y)

    def Predict(self, X):
        val = self.Activation(X)
        if val >= 0.0:
            return 1
        else:
            return -1

    def PrintModel(self, comment=""):
        print("======Model (" + comment + "):====")
        if self.std_features_:
            print("Standardize: mean: " + str(self.mean_) + " stddev: " + str(self.stddev_))
        print("Model weights: " + str(self.w_))
        print("===================================\n")

    # what is the accuracy of the current model
    # wrt input X and class labels y
    def Accuracy(self, X, y):
        if self.std_features_:
            X = self.Standardize(X)
        return self.AccuracyInternal(X, y)
    

    ######################
    # Private functions 
    ######################    
    def FitInternal(self, X, y):
        nsamples = X.shape[0]
        nfeatures = X.shape[1]
        print("X has shape " + str(np.shape(X)))
        print("y has shape " + str(np.shape(y)))
        # Randomize 1 + nfeatures -- the extra 1 is for the constant so we can assume threshold
        # is 0
        self.w_ = np.random.RandomState().normal(loc = 0.0, scale = 0.01, size = 1 + nfeatures)
        print("weights initialized -- shape is " + str(np.shape(self.w_)))
        self.PrintModel(comment = "random initialized")
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
        print("Training complete. Classification errors in each round"
              + "(out of " + str(nsamples) + "):")
        print(str(self.errors_))

    
    def NetInput(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def Activation(self, X):
        return self.NetInput(X)

    def CalcMeanAndStdDev(self, X):
        if self.mean_ is not None or self.stddev_ is not None:
            # We will calculate mean and stddev only once
            print("Likely error -- tried to call calc mean / std more than once")
            return

        mean_arr = []
        stddev_arr = []
        for i in range(X.shape[1]):
            mean = X[:, i].mean()
            mean_arr.append(mean)
            stddev = X[:, i].std()
            stddev_arr.append(stddev)
        # Save the mean and stddev - will be needed for converting 
        self.mean_ = np.array(mean_arr)
        self.stddev_ = np.array(stddev_arr)
        
    # Make each feature of a dataset have mean 0 and stddev 1
    def Standardize(self, X):
        if self.mean_ is None or self.stddev_ is None:
            # This means that this is the first time we are seeing data
            # Calculate the mean and standard deviation
            self.CalcMeanAndStdDev(X)
            return self.Standardize(X)
        # This assumes that self.mean_ and self.stddev_ are defined
        X_std = np.copy(X)
        for i in range(X.shape[1]):
            X_std[:, i] = (X[:, i] - self.mean_[i]) / self.stddev_[i]

        print("Standardized X")
        #print(X_std)
        return X_std

    # Calculate accuracy on X -- any standardization is assumed to have
    # been taken care of outside of this
    def AccuracyInternal(self, X, y):    
        errors = 0
        error_samples = []
        num = 0
        for xi, target in zip(X, y):
            if target != self.Predict(xi):
                errors += 1
                error_samples.append(num)
            num += 1
        nsamples = X.shape[0]
        ncorrect = nsamples - errors
        print("accuracy = " + str(ncorrect) + " / " + str(nsamples))
        print("errors = " + str(error_samples))
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
    p = Perceptron(100, 0.1)
    p.Fit(nX, ny)
    p.PrintModel(comment = "final model")
    p.Accuracy(nX, ny)

        
def main():
    # Run test
    test()


        
if __name__ == "__main__":
    # execute only if run as a script
    main()
