from perceptron import Perceptron
import numpy as np


class Adaline(Perceptron):
    """Adaline is a neural network which is like a perceptron with 2 differences both
       - The fitting is done wrt to the output of the activation function (real values), 
         not the final (binary) classification values
       - In Adaline, changes to weights depends on all samples. That is
         w --sample1, sample2, ..., sample n--> w' --sample1, sample2, ..., sample n---> w''
    """
    def Fit(self, X, y):
        nsamples = X.shape[0]
        nfeatures = X.shape[1]
        self.cost_ = []
        
        # Randomize 1 + nfeatures -- the extra 1 is for the constant so we can assume threshold
        # is 0
        self.w_ = np.random.RandomState().normal(loc = 0.0, scale = 0.01, size = 1 + nfeatures)
        self.PrintModel()
        self.Accuracy(X, y)

        # Iterate several times
        for _ in range(self.n_iter_):
            print("==============next iteration======================")
            # 1. find activation output
            output = self.Activation(X)
            print("output calculated - shape is " + str(np.shape(output)))
            # 2. find errors
            errors = y - output
            print("errors calculated - shape is " + str(np.shape(errors)))
            # 3. find weight delta
            w_del = X.T.dot(errors)
            print("weight delta calculated: " + str(w_del))
            # 4. Update weights
            self.w_[1:] += self.eta_ * w_del
            self.w_[0] += self.eta_ * errors.sum()
            # 5. Bookkeeping and logging
            # sum of squares errors
            cost = (errors**2).sum() / 2.0
            print("sum of errors squared = " + str(cost))
            self.cost_.append(cost)
            self.PrintModel()
            self.Accuracy(X, y)


