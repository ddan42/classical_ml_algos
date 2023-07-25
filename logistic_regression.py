import numpy as np
import pandas as pd
import random

class MyLogReg():
    def __init__(self, n_iter = 10, learning_rate = 0.1, weights = None, metric = None, reg = None, l1_coef = 0.0, l2_coef = 0.0, sgd_sample = None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.reg = reg
        if self.reg is None:
            self.l1_coef = 0.0
            self.l2_coef = 0.0
        elif self.reg == 'l1':
            self.l1_coef = l1_coef
            self.l2_coef = 0.0
        elif self.reg == 'l2':
            self.l1_coef = 0.0
            self.l2_coef = l2_coef
        elif self.reg == 'elasticnet':
            self.l1_coef = l1_coef
            self.l2_coef = l2_coef
    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    def __repr__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    @staticmethod    
    def get_confusion_dict(y_true, y_proba):
        y_pred = np.where(y_proba>0.5, 1, 0)
        confusion_dict = {'TP': 0, 'TN': 0, 'FN': 0, 'FP': 0}
        confusion_dict['TP'] = np.where((y_true == 1) & (y_pred == 1), 1, 0).sum()
        confusion_dict['FP'] = np.where((y_true == 0) & (y_pred == 1), 1, 0).sum()
        confusion_dict['FN'] = np.where((y_true == 1) & (y_pred == 0), 1, 0).sum()
        confusion_dict['TN'] = np.where((y_true == 0) & (y_pred == 0), 1, 0).sum()
        return confusion_dict
    def get_metric(self, y_true, y_proba):
        confusion_dict = self.get_confusion_dict(y_true, y_proba)
        if self.metric == 'accuracy':
            return (confusion_dict['TP']+confusion_dict['TN'])/(confusion_dict['TP']+confusion_dict['TN']+confusion_dict['FP']+confusion_dict['FN'])
        if self.metric == 'precision':
            return confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FP'])        
        if self.metric == 'recall':
            return confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FN'])
        if self.metric == 'f1':
            precision = confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FP'])        
            recall = confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FN'])
            return (2*precision*recall)/(precision+recall)
        if self.metric == 'roc_auc':
            y_proba = np.round(y_proba, 10)
            index_order = y_proba.argsort()[::-1]
            y_true = y_true[index_order]
            y_proba = y_proba[index_order]
            df = pd.DataFrame({'proba': y_proba, 'class': y_true})
            total = 0
            for idx, row in df.iterrows():
                if row['class']==0:
                    total += ((df['class']==1) & (df['proba'] > row['proba'])).sum() 
                    total += 0.5*((df['class']==1) & (df['proba'] == row['proba'])).sum()
            return total/((df['class']==1).sum()*(df['class']==0).sum())
    def fit(self, X, y, verbose = 0):
        def log_loss(y_true, y_pred):
            eps = 1e-15
            return -np.average(y_true@np.log(y_pred+eps)+(1-y_true)*np.log(1-y_pred+eps))+self.l1_coef*np.sum(np.abs(self.weights))+self.l2_coef*np.sum(self.weights**2)
        random.seed(self.random_state)
        X.insert(0, 'w0', 1)
        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        if self.sgd_sample is None:
            for i in range(self.n_iter):
                y_pred = 1/(1+np.exp(-np.dot(X, self.weights)))
                grad_logloss = np.dot((y_pred-y), X)/X.shape[0]+self.l1_coef*np.sign(self.weights)+2*self.l2_coef*self.weights
                if callable(self.learning_rate):
                    self.weights -= self.learning_rate(i+1)*grad_logloss
                else:
                    self.weights -= self.learning_rate*grad_logloss
                if (self.metric is not None) and (verbose != 0 and (i+1)%verbose==0):
                    print(f'{i+1}| loss: {log_loss(y, y_pred)}, {self.metric}: {self.get_metric(y, y_pred)}')  
        else:
            if type(self.sgd_sample) is float:
                self.sgd_sample=int(np.rint(self.sgd_sample*X.shape[0]))
            for i in range(self.n_iter):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                x_batch = X.iloc[sample_rows_idx]
                y_batch = y.iloc[sample_rows_idx]
                y_pred_batch = 1/(1+np.exp(-np.dot(x_batch, self.weights)))
                grad_logloss = np.dot((y_pred_batch-y_batch), x_batch)/x_batch.shape[0]+self.l1_coef*np.sign(self.weights)+2*self.l2_coef*self.weights
                if callable(self.learning_rate):
                    self.weights -= self.learning_rate(i+1)*grad_logloss
                else:
                    self.weights -= self.learning_rate*grad_logloss
                if (self.metric is not None) and (verbose != 0 and (i+1)%verbose==0):
                    print(f'{i+1}| loss: {log_loss(y_batch, y_pred_batch)}, {self.metric}: {self.get_metric(y_batch, y_pred_batch)}')  
        y_proba = 1/(1+np.exp(-np.dot(X, self.weights)))
        self.best_score = self.get_metric(y, y_proba)
    def predict_proba(self, X_pred):
        X_pred.insert(0, 'w0', 1)
        return 1/(1+np.exp(-np.dot(X_pred, self.weights)))
    def predict(self, X_pred):
        y_pred = self.predict_proba(self, X_pred)
        return np.where(y_pred>0.5, 1, 0)
    def get_coef(self):
        return self.weights[1:]
    def get_best_score(self):
        return self.best_score