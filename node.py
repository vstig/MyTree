from utils import get_candidate_splits, get_best_split, split_data
import numpy as np


class Node(object):
    def __init__(self):
        self.feature_split = None
        self.feature_value = None

        self.avg = float('inf')
        self.n_samples = 0

        self.left_child = None
        self.right_child = None

        self.is_terminal = False

    def __repr__(self):
        return ("feature: {}\n"
                "split value: {}\n"
                "# Samples: {}\n"
                "Avg y: {:0.3f}"
        ).format(self.feature_split, self.feature_value, self.n_samples, self.avg)

    def split_data(self, X, y):
        """
        :param X: dataframe of features
        :param y: column name of feature we are trying to predict
        :return: X_i, X_j, which is X split into 2 samples so as to maximize variance reduction
        """
        self.avg = X[y].mean()
        self.n_samples = X.shape[0]

        if X.drop_duplicates().shape[0] == 1:
            X_i, X_j = None, None

        else:
            candidate_splits = get_candidate_splits(X, y)
            self.feature_split, self.feature_value = get_best_split(X, candidate_splits, y)

            X_i, X_j = split_data(X, self.feature_split, self.feature_value)

        return X_i, X_j

    def which_branch(self, X):
        """
        :param X: New sample to pass through the split point
        :return: 'left' or 'right' indicating decision at split point
            if feature is numeric: left means X feature value is < node split value
                                   right means X feature >= node split
            if feature is boolean: left means X[node.feature] is False, right -> True
            if feature is category: left means X[node.feature] != node.feature_value
        """
        if self.feature_split is None:
            return "Node has not generated split"
        if np.issubdtype(type(X[self.feature_split]), np.number):
            if X[self.feature_split] < self.feature_value:
                return 'left'
            else:
                return 'right'
        elif np.issubdtype(type(X[self.feature_split]), np.bool_):
            if X[self.feature_split]:
                return 'right'
            else:
                return 'left'
        else:
            if X[self.feature_split] != self.feature_value:
                return 'left'
            else:
                return 'right'