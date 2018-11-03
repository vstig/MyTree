import numpy as np
import pandas as pd


def clean_and_backfill_data(df):
    for col in df:
        if set(df[col].unique()) == {'yes', 'no'}:
            df[col] = df[col].map({'yes': True, 'no': False})

    for col, dtype in df.dtypes.iteritems():
        if np.issubdtype(dtype, np.number):
            # If numeric, discretize cutpoints to be along the deciles (not a general rule, my simplification here)
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def get_dummie_data(df):
    for col in df.columns:
        if not (np.issubdtype(df[col].dtype, np.number) or np.issubdtype(df[col].dtype, np.bool_)):
            if set(df[col].unique()) == {'yes', 'no'}:
                df[col] = df[col].map({'yes': True, 'no': False})
            else:
                df = pd.merge(df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col),
                              left_index=True, right_index=True)
    return df