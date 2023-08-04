import pandas as pd
import numpy as np
import random

class MyTreeReg():
    def __init__(self, max_depth = 5, min_samples_split = 2,
                  max_leaves = 20, bins = None) -> None:
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
    def fit(self, X, y, N):
        self.fi = {column: 0 for column in X}
        self.N = N
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
    def predict(self, X_pred):
        self.preds = list()
        for idx, row in X_pred.iterrows():
            self.iterate_prediction(row, self.tree_dict)
        return self.preds
    
class MyForestReg():
    def __init__(self, n_estimators: int = 10, max_features: int = 0.5, max_samples: int = 0.5, 
                 max_depth: int = 5, min_samples_split: int = 2, max_leaves: int = 20, 
                 bins: int =16, oob_score = None, random_state: int = 42) -> None:
        """
        Random Forest Regression
        
        Random Forest Regression with customizable tree structures, feature importance and 
        Out of Bag score. Information gain, feature importance are calculated on MSE score.
        For OOB a few metrics can be specified.

        Parameters
        ----------
        n_estimators: int
            Number of trees in the forest.
        max_features: float
            Proportion of features (columns) each tree will get for training.
        max_samples: float
            Proportion of dataset (rows) each tree will get for training.
        max_depth: int
            A Decision Tree's parameter. A number which defines maximal 
            number of tree level. On maximum values terminal
            nodes (leaves) will be calculated.
        min_samples_splits: int
            A Decision Tree's parameter. A number which defines minimal number 
            of recordings which is needed to split a node into 2. 
            If number is less, a terminal node will be 
            constructed.
        max_leaves: int
            A Decision Tree's parameter.  The total number of leaves in a tree. 
            Tree preserves logical structure; if number is less than 2, 
            2 leaves will be created. If total number of leaves
            exceeds this number, growth will stop.
        bins: int
            A Decision Tree's parameter. Number of splits a tree will have. 
            If the number of unique values of each feature is more 
            than this number, bins will be created with np.histogram.
        oob_score: None or str
            If the metric is specified, Out of Bag score will be calculated in
            forest's self.oob_score_ variable. Can be one of 'mse', 'mae',
            'rmse', 'mape', 'r2'.
        random_state: int
            Seed for columns and rows selection for trees.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaves = max_leaves
        self.bins = bins
        self.random_state = random_state
        self.forest_dict = dict()
        self.fi = dict()
        self.oob_dict = dict()
        self.oob_score = oob_score
        self.oob_score_ = 0
        self.leaves_cnt = 0
    def __str__(self) -> str:
        return f'MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}, bins={self.bins}, random_state={self.random_state}'
    def __repr__(self) -> str:
        return f'MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}, bins={self.bins}, random_state={self.random_state}'
    def get_oob(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if self.oob_score == 'mse':
            return np.average((y_pred-y_true)**2)
        elif self.oob_score == 'mae':
            return np.average(np.abs(y_true-y_pred))
        elif self.oob_score == 'rmse':
            return np.average((y_pred-y_true)**2)**0.5
        elif self.oob_score == 'mape':
            return 100*np.average(np.abs(((y_true-y_pred)/y_true)))
        elif self.oob_score == 'r2':
            return 1-(np.sum((y_true-y_pred)**2))/(np.sum((y_true-np.average(y_true))**2))
    def fit(self, X: pd.DataFrame(dtype='object'), y:pd.DataFrame(dtype='object')) -> None:
        """
        Training loop of Random Forest

        During the training, instance iteratively creates Decision Trees which get
        random columns and rows for training, trees' feature importance is calculated
        by taking the number of rows of the whole dataset (forest's dataset). Trees' 
        structures are saved in dictionary. If oob_score is specified, the tree makes
        predictions on untrained rows. After training all trees,
        means are averaged and metric score is calculated.

        Parameters
        ----------
        X: pd.DataFrame()
            A dataframe with features for training.
        y: pd.Series()
            A series with target classes.
        """
        random.seed(self.random_state)
        init_cols = list(X.columns)
        n_cols = len(init_cols)
        init_rows_cnt = X.shape[0]
        cols_smpl_cnt = int(np.rint(self.max_features*n_cols))
        rows_smpl_cnt = int(np.rint(self.max_samples*init_rows_cnt))
        self.fi = {column: 0 for column in X}
        self.tree_fis = {}
        self.oob_dict = {idx: [] for idx in X.index}
        for n in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(X.index.to_list(), rows_smpl_cnt)
            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leaves, self.bins)
            # we need to calculate fi on the total number of rows. So trees should get
            # number of rows in the original dataset.
            tree.fit(X.loc[rows_idx][cols_idx], y.loc[rows_idx], len(y))
            self.forest_dict[n] = tree.tree_dict
            self.leaves_cnt += tree.leaves_cnt
            self.tree_fis[n] = tree.fi
            if self.oob_score:
                oob_idx = X[~X.index.isin(rows_idx)].index
                oob_X = X[cols_idx].loc[oob_idx]
                oob_preds = tree.predict(oob_X)
                for idx, pred in zip(oob_idx, oob_preds):
                    self.oob_dict[idx].append(pred)
        if self.oob_score:
            y_in_oob = list()
            y_pred_oob = list()
            for idx, lst in self.oob_dict.items():
                if lst:
                    y_in_oob.append(y.loc[idx])
                    y_pred_oob.append(np.mean(lst))
            self.oob_score_ = self.get_oob(y_in_oob, y_pred_oob)
        for tree_fi in self.tree_fis.values():
            for column, fi in tree_fi.items():
                self.fi[column] += fi
    def predict_row(self, row, tree):
        column, value = tree['column'], tree['value']
        if column not in ['leaf_left', 'leaf_right']:
            if row[column] <= value:
                side = 'left'
            elif row[column] > value:
                side = 'right'
            return self.predict_row(row, tree[side])
        else:
            return value
    def predict(self, X_pred: pd.DataFrame(dtype='object')):
        """
        A Random Forest's prediction based on averages of all trees

        Takes a dataframe with features, iteratively predicts mean for each from leaves of
        trees. Function outputs np.array of all objects' value.
        """
        predictions = pd.DataFrame()
        for idx, row in X_pred.iterrows():
            for tree_idx, tree in self.forest_dict.items():
                predictions.at[idx, f'tree_{tree_idx}'] = self.predict_row(row, tree)
        predictions = predictions.mean(axis = 1)
        return np.array(predictions)