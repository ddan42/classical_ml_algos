import numpy as np
import pandas as pd

class MyTreeReg():
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, 
                 max_leaves: int = 20, bins = None) -> None:
        """
        Regression Tree

        Regression Tree realization with customisable tree structures. Bins (histogram)
        of features is realized, but is designed for use in Random Forest 
        and Gradient Boostings.

        Parameters
        ----------

        max_depth: int
            A number which defines maximal number of tree level. On maximum values terminal
            nodes (leaves) will be calculated
        min_samples_split: int 
            A number which defines minimal number of recordings which is needed
            to split a node into 2. If number is less, a terminal node will be 
            constructed.
        max_leaves: int
            The total number of leaves in a tree. Tree preserves logical structure; if
            number is less than 2, 2 leaves will be created. If total number of leaves
            exceeds this number, growth will stop.
        bins: int or None
            Number of splits a tree will have. If not specified, tree's unique values
            will be used. If the number of unique values of each feature is more 
            than this number, bins will be created with np.histogram.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if max_leaves >= 2:
            self.max_leaves = max_leaves
        else:
            self.max_leaves = 2
        self.bins = bins
        self.histogram = dict()
        self.fi = dict()
        self.leaves_cnt = 0
    def __str__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}'
    def __repr__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}'
    def get_mse(self, y):
        return np.mean((y-np.mean(y))**2)
    def get_gain(self, s_0, y_left, y_right, n_left, n_right):
        c_left = n_left/(n_left+n_right)
        c_right = n_right/(n_left+n_right)
        return s_0 - (c_left*self.get_mse(y_left)+c_right*self.get_mse(y_right))
    def get_gain_by_split(self, X, y, column, split_value):
        s_0 = self.get_mse(y)
        y_left = y[X[column] <= split_value]
        y_right = y[X[column] > split_value]
        n_left = len(y_left)
        n_right = len(y_right)
        return self.get_gain(s_0, y_left, y_right, n_left, n_right)
    def get_best_split(self, X, y):
        column_split_gain = dict()
        column_splits = dict()
        for column in X:
            if self.bins is None:
                uniques = np.unique(X[column])
                column_splits[column] = [(uniques[i]+uniques[i+1])/2 for i in range(0, len(uniques)-1)]
            else:
                column_splits[column] = self.histogram[column]
            for split in column_splits[column]:
                gain = self.get_gain_by_split(X, y, column, split)
                if not np.isnan(gain):
                    column_split_gain[(column, split)] = self.get_gain_by_split(X, y, column, split)
        mx_key = max(column_split_gain, key = column_split_gain.get)
        return mx_key[0], mx_key[1]
    def node_test(self, y, depth):
        if len(y) == 1:
            return False
        if len(y) < self.min_samples_split:
            return False
        if depth == self.max_depth:
            return False
        if self.leaves_cnt == self.max_leaves:
            return False
        else:
            return True
    def iterate(self, X, y, depth, spliters, idx):
        self.leaves_cnt += 2
        Xs, ys = X.loc[idx], y.loc[idx]
        column, split_value = self.get_best_split(Xs, ys)
        tree_dict = {'column': column,
                     'value': split_value,
                     'left': None, 
                     'right': None}
        spliters.append([' ' * depth, depth, column, split_value])
        left_sub = Xs[Xs[column] <= split_value].index
        right_sub = Xs[Xs[column] > split_value].index
        yl = y.loc[left_sub]
        yr = y.loc[right_sub]
        N_p, N_l, N_r = len(ys), len(yl), len(yr)
        I, I_l, I_r = self.get_mse(ys), self.get_mse(yl), self.get_mse(yr)
        self.fi[column] += (N_p/self.N)*(I-(N_l/N_p)*I_l-(N_r/N_p)*I_r)
        depth += 1
        if self.node_test(yl, depth):
            self.leaves_cnt -= 1
            tree_dict['left'] = self.iterate(X, y, depth, spliters, left_sub)
        elif not self.node_test(yl, depth):
            spliters.append([' ' * depth, depth, 'leaf_left', np.mean(yl)])
            tree_dict['left'] = {'column': 'leaf_left', 'value': np.mean(yl)}
        if self.node_test(yr, depth):
            self.leaves_cnt -= 1
            tree_dict['right'] = self.iterate(X, y, depth, spliters, right_sub)
        elif not self.node_test(yr, depth):
            tree_dict['right'] = {'column': 'leaf_right', 'value': np.mean(yr)}
            spliters.append([' ' * depth, depth, 'leaf_right', np.mean(yr)])
        self.spliters = spliters
        return tree_dict
    def fit(self, X: pd.DataFrame(dtype='object'), y: pd.Series(dtype='object')):
        """
        Training of a tree.

        A model grows leaf-wise by minizing MSE for each split. It will recursively 
        iterate over the rows by first growing until reaching the most left 
        terminal node, then taking the right leaf of the node, etc. A leaf is a mean of its elements.
        
        Parameters
        ----------
        X: pd.DataFrame()
            A dataframe with features for training
        y: pd.Series()
            A series with target value
        """
        self.fi = {column: 0 for column in X}
        self.N = len(y)
        if self.bins is not None:
            for column in X:
                uniques = np.unique(X[column])
                splits = [(uniques[i]+uniques[i+1])/2 for i in range(0, len(uniques)-1)]
                if len(splits) <= self.bins-1:
                    self.histogram[column] = splits
                else:
                    self.histogram[column] = np.histogram(uniques, self.bins)[1][1:-1]
        self.tree_dict = self.iterate(X, y, depth = 0, spliters = list(), idx = X.index)
    def print_tree(self):
        """This function will print all split values used to build a tree."""
        for split in self.spliters:
            print(split[0] + split[2] + ' ' + str(split[3]))
    def iterate_prediction(self, row, tree):
        column, value = tree['column'], tree['value']
        if column not in ['leaf_left', 'leaf_right']:
            if row[column] <= value:
                self.iterate_prediction(row, tree['left'])
            elif row[column] > value:
                self.iterate_prediction(row, tree['right'])
        else:
            self.preds.append(value)
    def predict(self, X_pred: pd.DataFrame(dtype='object')):
        """Function will iterate each row until it falls into one leaf.
        Each leaf is the mean of its elements."""
        self.preds = list()
        for idx, row in X_pred.iterrows():
            self.iterate_prediction(row, self.tree_dict)
        return self.preds