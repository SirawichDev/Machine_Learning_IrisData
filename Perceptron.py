import numpy as np

class Perceptron(object):
    """
    Parameters
    eta : float
        Learning rate (between 0.1 -> 1.0)
    n_iter : int (จำนวนรอบ)
    random_state : int
    Random number generator seed for random weight<สุ่มเลขแล้ว mem ไว้>

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications (updates) in each epoch
    """

    def __init__(self, eta=0.01,n_iter = 50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        -------------
        X : (array-like), shape = [n_samples,n_features]
            training vectors, where n_samples is the number of
            samples and
            n_features is the number of features
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        --------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, size=1+X.shape[1])
