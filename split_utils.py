from operator import itemgetter
import numpy as np


def get_candidate_splits(X):
    """
    X: pandas dataframe of features, to search for possible splits
    """
    candidate_splits = []

    for col, dtype in X.dtypes.iteritems():
        if np.issubdtype(dtype, np.number):
            # If numeric, discretize cutpoints to be along the deciles (not a general rule, my simplification here)
            if X[col].nunique() < 5:
                candidate_splits += [(col, v) for v in sorted(X[col].unique())[1:]]
            else:
                candidate_splits += [(col, v) for v in set(np.percentile(X[col].dropna(), range(5, 100, 5)))]
        elif np.issubdtype(dtype, np.bool_):
            candidate_splits += [(col, True)]
        else:
            # If categorical, split on the 10 most frequenty occuring values (again my simplification)
            if X[col].nunique() < 10:
                candidate_splits += [(col, v) for v in X[col].unique()]
            else:
                candidate_splits += [(col, v) for v in X[col].value_counts().sort_values(ascending=False).index[:10]]

    return candidate_splits


def get_split_masks(X, feature, val):
    if np.issubdtype(X[feature].dtype, np.number):
        left_mask = (X[feature] < val)
        right_mask = (X[feature] >= val)
    else:
        left_mask = (X[feature] != val)
        right_mask = (X[feature] == val)

    return left_mask, right_mask


def split_data(X, feature, val):
    left_mask, right_mask = get_split_masks(X, feature, val)

    X_i, X_j = X.loc[left_mask], X.loc[right_mask]

    return X_i, X_j


def get_split_variance(X, feature, val, y):
    left_mask, right_mask = get_split_masks(X, feature, val)

    y_i, y_j = y.loc[left_mask], y.loc[right_mask]

    return ((np.var(y_i) * len(y_i)) + (np.var(y_j) * len(y_j))) / len(y)


def get_split_stats(X, candidate_splits, y):
    variance_reductions = []
    for (feature, value) in candidate_splits:
        split_variance = get_split_variance(X, feature, value, y)
        left_branch, right_branch = get_split_masks(X, feature, value)
        # TO-DO: Proactively stop this from happening
        if not np.isnan(split_variance):
            split_info = {'feature': feature,
                          'value': value,
                          'variance_reduction': np.var(y) - split_variance,
                          'n_samples_left': left_branch.sum(),
                          'n_samples_right': right_branch.sum()}
            variance_reductions.append(split_info)

    variance_reductions.sort(key=itemgetter('variance_reduction'), reverse=True)

    return variance_reductions


def get_valid_splits(X, candidate_splits, y, min_reduction=0, min_samples_leaf=1):
    return [split for split in get_split_stats(X, candidate_splits, y)
            if (split['variance_reduction'] > min_reduction and
                min(split['n_samples_left'], split['n_samples_right'])>=min_samples_leaf)
            ]
