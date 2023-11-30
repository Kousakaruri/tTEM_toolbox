import pandas as pd


def compatibility_search(fname, pattern):
    if isinstance(fname, dict):
        keys = [str(key).strip() for key in list(fname.keys())]
        match = [key for key in keys if key in pattern]
        return match
    elif isinstance(fname, pd.DataFrame):
        columns = [str(key).strip() for key in list(fname.columns)]
        match = [column for column in columns if column in pattern]
        return match
    else:
        raise TypeError('fname must be dict or DataFrame')
