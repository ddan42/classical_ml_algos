import numpy as np
import pandas as pd
import random

class MyLineReg():
    def __init__(self, n_iter = 100, learning_rate = 0.1, weights = None, 
                 metric = 'mse', reg = None, l1_coef=0, l2_coef=0, 
                 sgd_sample = None, random_state = 42) -> None:
        '''
        Linear regression with regularization and Gradient Descent.
        
        Linear regression algorithm with Gradient Descent, regularisation and 
        Stochastic Gradient Descent. 
        Learning rate can be constant or scheduled on number of iterations with lambda function.
        Model can print several metrics (mse, mae, etc) and optimizes MSE loss function.

        Parameters
        ----------
        
        n_iter: int
            Number of iterations of updating weights based on loss function.
        learning_rate: float or lambda
            A factor by which gradients of weights are multiplied at each iteration.
            It might be a function of iteration.
        weights: None
            Weights that model will use for predicting target. They are initialized and
            update during training.
        metric: str
            A metric which will be printed during fitting.
        reg: None or str
            None if no regularization is used. 'l1' for l1-norm, 'l2' for l2-norm, 'elasticnet' for elastic net.
        l1_coef:
            A helper parameter for regularization.
        l2_coef:
            A helper parameter for regularization.
        sgd_sample: int or float
            A number or share of samples to use for Stochastic Gradient Descent.
            SGD will be used only if this value is specified.
        '''
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = str(metric)
        self.reg = reg
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        if self.reg is None:
            self.l1_coef = 0
            self.l2_coef = 0
        if reg == 'l1':
            self.l1_coef = l1_coef
            self.l2_coef = 0     
        if reg == 'l2':
            self.l1_coef = 0
            self.l2_coef = l2_coef
        if reg == 'elasticnet':
            self.l1_coef = l1_coef
            self.l2_coef = l2_coef
    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    def fit(self, X: pd.DataFrame(dtype = 'object'), y: pd.Series(dtype = 'object'), verbose = False) -> None: 
        """
        Fitting data.

        A model fits and trains a model. If verbose is specified, each number of steps from
        verbose, metric score will be printed.

        Parameters
        ----------
        X: pd.DataFrame()
            DataFrame with features. The model will copy data for training.
        y: pd.Series()
            Series with target values.
        verbose: None or int
            Verbose specifies number of steps at which metric score will be printed.
        """
        def get_mse(y_true, y_pred):
            return np.average((y_pred-y)**2)+self.l1_coef*np.sum(np.abs(self.weights))+self.l2_coef*np.sum(self.weights**2)
        def get_mae(y_true, y_pred):
            return np.average(np.abs(y_true-y_pred))
        def get_rmse(y_true, y_pred):
            return np.average((y_pred-y)**2)**0.5
        def get_mape(y_true, y_pred):
            return 100*np.average(np.abs(((y_true-y_pred)/y_true)))
        def get_r2(y_true, y_pred):
            return 1-(np.sum((y_true-y_pred)**2))/(np.sum((y_true-np.average(y_true))**2))
        metrics_dict = {'mse': get_mse, 
                        'mae': get_mae, 
                        'rmse': get_rmse, 
                        'mape': get_mape, 
                        'r2': get_r2
                        }
        self.X = X.copy()
        self.X.insert(0, 'ones', 1, allow_duplicates=True)
        n_features = len(self.X.columns)
        n_rows = len(self.X.index)
        self.weights = np.ones(n_features)
        self.weights_list = list()
        if self.sgd_sample is None:
            for i in range(self.n_iter):
                y_pred = np.dot(self.X, self.weights)
                grad_MSE = (2*(y_pred-y).dot(self.X))/(n_rows)+self.l1_coef*np.sign(self.weights)+2*self.l2_coef*self.weights
                if callable(self.learning_rate):
                    lr = self.learning_rate(i+1)
                    self.weights -= lr*grad_MSE
                else:
                    self.weights -= self.learning_rate*grad_MSE
                if verbose!=0 and ((i+1)%verbose==0 or i == 0):
                    print(f'{i+1}|loss: {metrics_dict[self.metric](y, y_pred)}')
        else: 
            random.seed(self.random_state)
            if type(self.sgd_sample) is float:
                self.sgd_sample=int(np.rint(self.sgd_sample*n_rows))
            for i in range(self.n_iter):
                sample_rows_idx = random.sample(range(n_rows), self.sgd_sample)
                x_batch = self.X.iloc[sample_rows_idx]
                n_rows_batch = len(x_batch.index)
                y_batch = y.iloc[sample_rows_idx]
                y_pred_batch = np.dot(x_batch, self.weights)
                grad_MSE = (2*(y_pred_batch-y_batch).dot(x_batch))/(n_rows_batch)+self.l1_coef*np.sign(self.weights)+2*self.l2_coef*self.weights
                if callable(self.learning_rate):
                    lr = self.learning_rate(i+1)
                    self.weights -= lr*grad_MSE
                else:
                    self.weights -= self.learning_rate*grad_MSE
                if verbose!=0 and ((i+1)%verbose==0 or i == 0):
                    print(f'{i+1}|loss: {metrics_dict[self.metric](y, np.dot(self.X, self.weights))}')
        self.final_score = metrics_dict[self.metric](y, np.dot(self.X, self.weights))
    def get_coef(self) -> list:
        """Returns all weights of linear model excluding bias as np.array."""
        return self.weights[1:]
    def predict(self, X_pred: pd.DataFrame(dtype = 'object')) -> np.array:
        """Predicts target values from a dataframe with features. The model copies 
        data for training."""
        self.X_pred = X_pred.copy()
        self.X_pred.insert(0, 'ones', 1, allow_duplicates=True)
        return np.dot(self.X_pred, self.weights)
    def get_best_score(self) -> float:
        """Returns score of a metric score achieved after the training."""
        return self.final_score