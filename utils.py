import pandas as pd
import numpy as np


def get_candidate_splits(df, y):
    """
    df: pandas dataframe, to search for possible splits
    y: column name of value we are trying to predict (so omit from candidate splits)
    """
    candidate_splits = []

    for col, dtype in df.dtypes.iteritems():
        if col == y:
            # We don't want to split on target variable
            continue
        if np.issubdtype(dtype, np.number):
            # If numeric, discretize cutpoints to be along the deciles (not a general rule, my simplification here)
            if df[col].nunique() < 5:
                candidate_splits += [(col, v) for v in sorted(df[col].unique())[1:]]
            else:
                candidate_splits += [(col, v) for v in set(np.percentile(df[col].dropna(), range(5, 100, 5)))]
        else:
            # If categorical, split on the 10 most frequenty occuring values (again my simplification)
            if df[col].nunique() < 10:
                candidate_splits += [(col, v) for v in df[col].unique()]
            else:
                candidate_splits += [(col, v) for v in df[col].value_counts().sort_values(ascending=False).index[:10]]

    return candidate_splits


def get_split_variance(X, feature, val, y):
    if np.issubdtype(X[feature].dtype, np.number):
        X_i = X.loc[X[feature] < val]
        X_j = X.loc[X[feature] >= val]
    else:
        X_i = X.loc[X[feature] != val]
        X_j = X.loc[X[feature] == val]

    return ((np.var(X_i[y]) * X_i.shape[0]) + (np.var(X_j[y]) * X_j.shape[0])) / X.shape[0]


def get_variance_reduction(df, candidate_splits, y):
    variance_reductions = []
    for split in candidate_splits:
        variance_reductions.append((split[0], split[1], get_split_variance(df, split[0], split[1], y)))

    var_df = pd.DataFrame(variance_reductions, columns=['feature', 'value', 'variance'])
    var_df['variance_reduction'] = df[y].var() - var_df['variance']

    return var_df.sort_values('variance_reduction', ascending=False)


def get_best_split(df, candidate_splits, y):
    return tuple(get_variance_reduction(df, candidate_splits, y).iloc[0][['feature', 'value']])


def split_data(X, feature, val):
    if np.issubdtype(X[feature].dtype, np.number):
        X_i = X.loc[X[feature] < val]
        X_j = X.loc[X[feature] >= val]
    else:
        X_i = X.loc[X[feature] != val]
        X_j = X.loc[X[feature] == val]

    return X_i, X_j
