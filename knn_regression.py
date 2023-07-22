import numpy as np
import pandas as pd

class MyKNNReg():
    def __init__(self, k = 3,  metric = 'euclidean', weight = 'uniform') -> None:
        """
        kNN algorithm for regression realisation with numpy and pandas.

        The realisation of kNN algorithm for regression task which predicts based on the mean 
        of neighbours. To count distances, the model can use euclidean, 
        manhattan, chebyshev or cosine distance. Weighted mean based on rank and distance 
        of neighbours are realised too. 
        
        Parameters
        ----------
        k : int
            number of closest neighbours to choose
        
        metric : str
            One of metrics to calculate the distance between objects: euclidean, 
            manhattan, chebyshev, cosine.
        
        weights : str
            One of type of weight to choose when calculating the average of neighbours: 
            uniform (mode), rank (inversely proportional to the rank of a neighbour 
            based on how close it is), distance (inversely proportional 
            to the distance between test and train row).
        """
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight
    def __str__(self) -> str:
        return f'MyKNNReg class: k={self.k}'
    def __repr__(self) -> str:
        return f'MyKNNReg class: k={self.k}'
    def fit(self, X: pd.DataFrame(dtype = 'object'), y: pd.Series(dtype = 'object')):
        """The model just saves the train data."""
        self.X = X
        self.y = y
        self.train_size = X.shape
    def get_distance(self, row: pd.Series(dtype = 'object')):
        if self.metric == 'euclidean':
            distances = ((self.X - row)**2).sum(axis=1)**0.5
        elif self.metric == 'manhattan':
            distances = (np.abs(self.X - row)).sum(axis=1)
        elif self.metric == 'chebyshev':
            distances = np.abs(self.X - row).max(axis=1)
        elif self.metric == 'cosine':
            X_copy = self.X.copy()
            scalars = X_copy @ row
            row_distance = 0.0
            for idx, value in row.iteritems():
                row_distance += value**2
                row_distance = row_distance**0.5
            X_copy['x_distance'] = (np.square(X_copy)).sum(axis=1)**0.5
            distances = 1 - (scalars/(X_copy['x_distance']*row_distance))
        return distances
    def get_weigted_mean(self, y_k: pd.Series(dtype = 'object'), distances_k: pd.Series(dtype = 'object')):
        if self.weight == 'uniform':
            return np.average(y_k)
        elif self.weight == 'rank':
            distances_k = distances_k.sort_values(ascending = True)
            y_k = y_k[distances_k.index]
            reverse_ranks = 1/distances_k.rank(ascending=True)
            weights = reverse_ranks/(reverse_ranks.sum())        
            return y_k@weights
        elif self.weight == 'distance': 
            distances_k = distances_k.sort_values(ascending = True)
            y_k = y_k[distances_k.index]
            reverse_distance = 1/distances_k
            weights = reverse_distance/(reverse_distance.sum())        
            return y_k@weights
    def predict(self, X_pred: pd.DataFrame(dtype = 'object')):
        """Model outputs neighbours' average or weighted average inversely proportional
        to rank or distance of each train row."""
        preds = list()
        for idx, row in X_pred.iterrows():
            distances = self.get_distance(row)
            distances_k = distances.nsmallest(self.k)
            y_k = self.y[distances_k.index]
            preds.append(self.get_weigted_mean(y_k, distances_k))
        return np.array(preds)