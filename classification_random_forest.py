import numpy as np
import pandas as pd
import random 

class MyTreeClf():
    def __init__(self, max_depth = 5, min_samples_split = 2, max_leaves = 20, 
                 bins = None, criterion = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if max_leaves >= 2:
            self.max_leaves = max_leaves
        else:
            self.max_leaves = 2
        self.bins = bins
        self.histogram = dict()
        self.criterion = criterion
        self.fi = dict()
        self.leaves_cnt = 0
    def __str__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}'
    def __repr__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}'
    def get_criterion(self, y):
        s = 0
        m = len(y)
        unique_classes = np.unique(y)
        if self.criterion == 'entropy':
            for u_class in unique_classes:
                n = np.where(y==u_class, 1, 0).sum()
                if n != 0:
                    s += (n/m)*np.log2(n/m)
            return -s
        elif self.criterion == 'gini':
            for u_class in unique_classes:
                n = np.where(y==u_class, 1, 0).sum()
                s += (n/m)**2
            return 1-s
    def get_ig(self, s_0, n, n_left, n_right, s_left, s_right):
        return s_0 - (n_left/n)*s_left - (n_right/n)*s_right
    def ig_split(self, X, y, column, split):
        s_0 = self.get_criterion(y)
        y_left = y[X[column] <= split]
        y_left = y[X[column] <= split]
        y_right = y[X[column] > split]
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        s_left = self.get_criterion(y_left)
        s_right = self.get_criterion(y_right)
        return self.get_ig(s_0, n, n_left, n_right, s_left, s_right)
    def get_best_split(self, X, y):
        column_split_ig = dict()
        column_splits = dict()
        for column in X:
            if self.bins is None:
                uniques = np.unique(X[column])
                splits = [(uniques[i]+uniques[i+1])/2 for i in range(0, len(uniques)-1)]
                column_splits[column] = splits
            else:
                column_splits[column] = self.histogram[column]
            for split in column_splits[column]:
                column_split_ig[(column, split)] = self.ig_split(X, y, column, split)
        mx_key = max(column_split_ig, key = column_split_ig.get)
        col_name, split_value, ig = mx_key[0], mx_key[1], column_split_ig[mx_key]
        return col_name, split_value
    def node_test(self, y, depth):
        if len(y) == 1:
            return False
        if len(np.unique(y)) == 1:
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
        I, I_l, I_r = self.get_criterion(ys), 0, 0
        if self.node_test(yl, depth):
            I_l = self.get_criterion(yl)
        if self.node_test(yr, depth):
            I_r = self.get_criterion(yr)
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

class MyForestClf():
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, 
                 max_samples: float = 0.5, max_depth: int = 5,
                 min_samples_split: int = 2, max_leaves: int = 20, 
                 bins: int = 16, criterion: str = 'entropy', oob_score = None, 
                 random_state: int =42) -> None:
        """
        Random Forest Classification
        
        Random Forest Classification with customizable tree structures, Gini and Entropy
        as information gains, feature importance and Out of Bag score.

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
        criterion: str
            A Decision Tree's parameter. Default criterion is 'entropy', 
            but can be modified to use Gini - 'gini'.
        oob_score: None or str
            If the metric is specified, Out of Bag score will be calculated in
            forest's self.oob_score_ variable. Can be one of 'accuracy', 
            'precision', 'recall', 'f1' or 'roc_auc'.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaves = max_leaves
        self.bins = bins
        self.criterion = criterion
        self.random_state = random_state
        self.forest_dict = dict()
        self.fi = dict()
        self.oob_dict = dict()
        self.oob_score = oob_score
        self.oob_score_ = 0
        self.leaves_cnt = 0
    def __str__(self) -> str:
        return f'MyForestClf class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}, bins={self.bins}, criterion={self.criterion}, random_state={self.random_state}'
    def __repr__(self) -> str:
        return f'MyForestClf class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leaves={self.max_leaves}, bins={self.bins}, criterion={self.criterion}, random_state={self.random_state}'
    def get_oob(self, y_true, y_proba):
        confusion_dict = dict()
        y_pred = np.array([1 if y_p>0.5 else 0 for y_p in y_proba])
        y_true = np.array(y_true)
        confusion_dict = {'TP': 0, 'TN': 0, 'FN': 0, 'FP': 0}
        confusion_dict['TP'] = np.where((y_true == 1) & (y_pred == 1), 1, 0).sum()
        confusion_dict['FP'] = np.where((y_true == 0) & (y_pred == 1), 1, 0).sum()
        confusion_dict['FN'] = np.where((y_true == 1) & (y_pred == 0), 1, 0).sum()
        confusion_dict['TN'] = np.where((y_true == 0) & (y_pred == 0), 1, 0).sum()
        if self.oob_score == 'accuracy':
            return (confusion_dict['TP']+confusion_dict['TN'])/(confusion_dict['TP']+confusion_dict['TN']+confusion_dict['FP']+confusion_dict['FN'])
        elif self.oob_score == 'precision':
            return confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FP'])        
        elif self.oob_score == 'recall':
            return confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FN'])
        elif self.oob_score == 'f1':
            precision = confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FP'])        
            recall = confusion_dict['TP']/(confusion_dict['TP']+confusion_dict['FN'])
            return (2*precision*recall)/(precision+recall)
        elif self.oob_score == 'roc_auc':
            y_proba = np.round(y_proba, 10)
            index_order = y_proba.argsort()[::-1]
            y_true = y_true[index_order]
            y_proba = y_proba[index_order]
            df = pd.DataFrame({'proba': y_proba, 'class': y_true, 'sum': np.zeros(len(y_true))})
            total = 0
            for idx, row in df.iterrows():
                if row['class']==0:
                    total += ((df['class']==1) & (df['proba'] > row['proba'])).sum() 
                    total += 0.5*((df['class']==1) & (df['proba'] == row['proba'])).sum()
            return total/((df['class']==1).sum()*(df['class']==0).sum())
    def fit(self, X: pd.DataFrame(dtype='object'), y: pd.DataFrame(dtype='object')):
        """
        Training loop of Random Forest

        During the training, instance iteratively creates Decision Trees which get
        random columns and rows for training, trees' feature importance is calculated
        by taking the number of rows of the whole dataset (forest's dataset). Trees' 
        structures are saved in dictionary. If oob_score is specified,  probability
        for class 1 is calculated and stored a dataframe. After training all trees,
        probabilities are averaged and metric score is calculated.

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
        self.oob_dict = {idx: [] for idx in X.index}
        for n in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(X.index.to_list(), rows_smpl_cnt)
            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leaves, self.bins, self.criterion)
            # we need to calculate fi on the total number of rows. So trees should get
            # number of rows in the original dataset.
            tree.fit(X.loc[rows_idx][cols_idx], y.loc[rows_idx], len(y))
            self.forest_dict[n] = tree.tree_dict
            for feature, fi in tree.fi.items():
                self.fi[feature] += fi
            if self.oob_score:
                oob_idx = X[~X.index.isin(rows_idx)].index
                oob_X = X[cols_idx].loc[oob_idx]
                oob_preds = tree.predict(oob_X)
                for idx, pred in zip(oob_idx, oob_preds):
                    self.oob_dict[idx].append(pred)
            self.leaves_cnt += tree.leaves_cnt
        if self.oob_score:
            y_in_oob = list()
            y_proba_oob = list()
            for idx, lst in self.oob_dict.items():
                if lst:
                    y_in_oob.append(y.loc[idx])
                    y_proba_oob.append(np.mean(lst))
            self.oob_score_ = self.get_oob(y_in_oob, y_proba_oob)
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
    def predict(self, X_pred: pd.DataFrame(dtype='object'), type: str) -> np.array:
        """
        A Random Forest's prediction with vote type specification

        Takes dataframe with features, iteratively predicts probabilities of each class.
        'mean' or 'vote' are types of voting for class prediction. Function outputs 
        np.array of all objects' classes.

        Parameters
        ----------
        X_pred: pd.DataFrame
            A dataframe with features to predict
        type: str
            If 'mean' is passed as vote type, probabilities for each class 
            are averaged  and the class of threshold 0.5 if added to the list. 
            If 'vote' is passed as vote type, each probability is converted
            to class by threshold 0.5, the mode for each object is taken. If classes
            have the same frequencies, class 1 is returned.
        """
        predictions = pd.DataFrame()
        for idx, row in X_pred.iterrows():
            for tree_idx, tree in self.forest_dict.items():
                predictions.at[idx, f'tree_{tree_idx}'] = self.predict_row(row, tree)
        if type == 'mean':
            predictions = [1 if pred > 0.5 else 0 for pred in predictions.mean(axis = 1)]
        elif type == 'vote':
            predictions = predictions.applymap(lambda x: 1 if x > 0.5 else 0)
            predictions = predictions.apply(lambda x: 1 if np.where(x==1, 1, 0).sum() >= np.where(x==0, 1, 0).sum() else 0, axis=1)
        return np.array(predictions)
    def predict_proba(self, X_pred: pd.DataFrame(dtype='object')) -> np.array:
        """
        Takes dataframe with features, iteratively predicts probabilities of each class
        and returns the average for each object as np.array().
        """
        predictions = pd.DataFrame()
        for idx, row in X_pred.iterrows():
            for tree_idx, tree in self.forest_dict.items():
                predictions.at[idx, f'tree_{tree_idx}'] = self.predict_row(row, tree)
        return np.array(predictions.mean(axis = 1).to_list())