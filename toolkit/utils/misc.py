import pandas as pd


def pd_merge_differences(df1: pd.DataFrame, df2: pd.DataFrame, keep_left: bool = False, **kwargs) -> pd.DataFrame:
    """Merge two dataframes without duplicating columns
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Dataframes to merge.
    keep_left : bool, optional
        Whether to keep the columns from the left dataframe
        or otherwise the columns from the right dataframe.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `pandas.merge` function.
    """
    if keep_left:
        cols_to_keep = [col for col in df2.columns if col not in df1.columns]
        df2 = df2[cols_to_keep]
    else:
        cols_to_keep = [col for col in df1.columns if col not in df2.columns]
        df1 = df1[cols_to_keep]
    return pd.merge(df1, df2, **kwargs)

