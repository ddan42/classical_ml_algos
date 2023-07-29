import numpy as np
import pandas as pd
import random

import multiprocessing
from multiprocessing import Pool, Manager

from classification_random_forest import MyForestClf, MyTreeClf
    
class MultiProcessForestClf(MyForestClf):

    @staticmethod
    def tree_fit(self, n, X, y, cols_idx, rows_idx):
        tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leaves, self.bins, self.criterion)
        # we need to calculate fi on the total number of rows. So trees should get
        # number of rows in the original dataset.
        tree.fit(X.loc[rows_idx][cols_idx], y.loc[rows_idx], len(y))

        self.m_forest_dict[n] = tree.tree_dict
        self.m_tree_fi[n] = tree.fi
        self.m_leaves[n] = tree.leaves_cnt

        if self.oob_score:
            oob_idx = X[~X.index.isin(rows_idx)].index
            oob_X = X[cols_idx].loc[oob_idx]
            oob_preds = tree.predict(oob_X)
            self.oob_df[n] = [oob_idx, oob_preds]
    
    def fit(self, X: pd.DataFrame(dtype='object'), y: pd.DataFrame(dtype='object')):
        random.seed(self.random_state)
        init_cols = list(X.columns)
        n_cols = len(init_cols)
        init_rows_cnt = X.shape[0]
        cols_smpl_cnt = int(np.rint(self.max_features*n_cols))
        rows_smpl_cnt = int(np.rint(self.max_samples*init_rows_cnt)) 

        self.fi = {column: 0 for column in X}

        cols_idx_s = []
        rows_idx_s = []
        for n in range(self.n_estimators):
            cols_idx_s.append(random.sample(init_cols, cols_smpl_cnt))
            rows_idx_s.append(random.sample(X.index.to_list(), rows_smpl_cnt))

        manager = Manager()
        self.m_tree_fi = manager.dict()
        self.oob_df = manager.dict()
        self.m_forest_dict = manager.dict()
        self.m_leaves = manager.dict()
        
        pool = Pool(multiprocessing.cpu_count())
        pass_lst = [[self, n, X, y, cols_idx_s[n], rows_idx_s[n]] for n in range(self.n_estimators)]

        results = pool.starmap(self.tree_fit, pass_lst)
        pool.close()        


        for i in range(self.n_estimators):
            for column, fi in self.m_tree_fi[i].items():
                self.fi[column] += fi
        
        if self.oob_score:
            self.oob_dict = {i: [] for i in X.index}

            for i in range(self.n_estimators):
                for idx, pred in zip(self.oob_df[i][0], self.oob_df[i][1]):
                    self.oob_dict[idx].append(pred)

            y_in_oob = list()
            y_proba_oob = list()
            for idx, lst in self.oob_dict.items():
                if lst:
                    y_in_oob.append(y.loc[idx])
                    y_proba_oob.append(np.mean(lst))
            self.oob_score_ = self.get_oob(y_in_oob, y_proba_oob)

        self.forest_dict = self.m_forest_dict.copy()
        self.leaves_cnt = sum(self.m_leaves.values())