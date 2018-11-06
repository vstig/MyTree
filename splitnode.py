from split_utils import get_candidate_splits, get_valid_splits, split_data
import numpy as np


class SplitNode(object):
    def __init__(self):
        self.feature_split = None
        self.feature_value = None

    def __repr__(self):
        return ("feature: {}\n"
                "split value: {}\n"
        ).format(self.feature_split, self.feature_value)

    def split_data(self, X, y, **kwargs):
        """
        :param X: dataframe of features
        :param y: column name of feature we are trying to predict
        optional kwargs: min_samples_leaf, min_reduction
        :return: X_i, X_j, which is X split into 2 samples so as to maximize variance reduction
        """
        all_splits = get_candidate_splits(X)
        valid_splits = get_valid_splits(X, all_splits, y, **kwargs)
        if valid_splits:
            best_split = valid_splits[0]
            self.feature_split, self.feature_value = best_split['feature'], best_split['value']

            X_i, X_j = split_data(X, self.feature_split, self.feature_value)

            return (X_i, y.loc[X_i.index]), (X_j, y.loc[X_j.index])
        else:
            return None

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