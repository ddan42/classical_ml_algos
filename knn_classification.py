import numpy as np
import pandas as pd

class MyKNNClf():
    """
    kNN alogithm's realisation in numpy and pandas.

    The relesation of kNN algorthm for classification task which can predict proba 
    (probability of class 1 in neighbors) and class (mdoe of neighbours). To conut distances, the model can use euclidean, 
    manhattan, chebyshev or cosine distance. Class prediction based on rank and distance of neighbours are realised too. 
    """
    def __init__(self, k = 3, metric = 'euclidean', weight = 'uniform') -> None:
        """
        Parameters
        ----------
        k : int
            number of closest neiughbours to choose
        
        metric : str
            One of metrics to calculate the distance between objects: euclidean, manhattan, chebyshev, cosine.
        
        weights : str
            One of type of weight to choose when calculating the average of neighbours: uniform (mode), rank (inversely proportional 
            to the rank of a neighbour based on how close it is), distance (inversely proportional to the distance between test
            and train row)
        """
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight
    def __str__(self) -> str:
        return f'MyKNNClf class: k={self.k}'
    def __repr__(self) -> str:
        return f'MyKNNClf class: k={self.k}'
    def fit(self, X: pd.DataFrame(dtype = 'object'), y: pd.Series(dtype = 'object')) -> None:
        """The model just saves the train data. """
        self.X = X
        self.y = y
        self.train_size = X.shape
    def get_distance(self, row: pd.Series(dtype = 'object')) -> float:
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
    def predict_get_pred_class(self, y_k: pd.Series(dtype = 'object'), distances_k: pd.Series(dtype = 'object')) -> int:
        if self.weight == 'uniform':
            ones = np.where(y_k == 1, 1, 0).sum()
            zeroes = np.where(y_k == 0, 1, 0).sum()
            if ones >= zeroes:
                return 1
            elif ones < zeroes:
                return 0
        elif self.weight == 'rank':
            reverse_ranks = 1/distances_k.rank(ascending=True)
            class_weights = dict()
            class_weights[1] = reverse_ranks[y_k[y_k==1].index].sum()/reverse_ranks.sum()
            class_weights[0] = reverse_ranks[y_k[y_k==0].index].sum()/reverse_ranks.sum()
            return max(class_weights, key = class_weights.get)
        elif self.weight == 'distance':
            reverse_distances = 1/(distances_k+1e-15)
            class_weights = dict()
            class_weights[1] = reverse_distances[y_k[y_k==1].index].sum()/reverse_distances.sum()
            class_weights[0] = reverse_distances[y_k[y_k==0].index].sum()/reverse_distances.sum()
            return max(class_weights, key = class_weights.get)
    def proba_get_pred_class(self, y_k: pd.Series(dtype = 'object'), distances_k: pd.Series(dtype = 'object')) -> float:
        if self.weight == 'uniform':
            return np.mean(y_k)
        elif self.weight == 'rank':
            reverse_ranks = 1/distances_k.rank(ascending=True)
            class_weights = dict()
            class_weights[1] = reverse_ranks[y_k[y_k==1].index].sum()/reverse_ranks.sum()
            class_weights[0] = reverse_ranks[y_k[y_k==0].index].sum()/reverse_ranks.sum()
            return class_weights[1]
        elif self.weight == 'distance':
            reverse_distances = 1/(distances_k+1e-15)
            class_weights = dict()
            class_weights[1] = reverse_distances[y_k[y_k==1].index].sum()/reverse_distances.sum()
            class_weights[0] = reverse_distances[y_k[y_k==0].index].sum()/reverse_distances.sum()
            return class_weights[1]
    def predict(self, X_pred: pd.DataFrame(dtype = 'object')) -> np.array:
        """Model outputs most popular class or highest probability class inversely proportional to rank or distance of each train row."""
        preds = list()
        for idx, row in X_pred.iterrows():
            distances = self.get_distance(row)
            distances_k = distances.nsmallest(self.k)
            y_k = self.y[distances_k.index]
            preds.append(self.predict_get_pred_class(y_k, distances_k))
        return np.array(preds)
    def predict_proba(self, X_pred: pd.DataFrame()) -> np.array:
        """Model outputs mean of neighbours or weighted mean of neighbours inveresely proportional to rank or distance of each train row."""
        preds = list()
        for idx, row in X_pred.iterrows():
            distances = distances = self.get_distance(row)
            distances_k = distances.nsmallest(self.k)
            y_k = self.y[distances_k.index]
            preds.append(self.proba_get_pred_class(y_k, distances_k))
        return np.array(preds)