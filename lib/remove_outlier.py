def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.01)
    q3 = df_in[col_name].quantile(0.99)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) &  (df_in[col_name] < fence_high)]
    return df_out

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv("../data/clearedData_dated.csv",index_col="timestamp",parse_dates=True)

    print(df["Last_Rechenwert"].describe())
    #skip the booleans for days
    for col in df.columns[:-8]:
        df = remove_outlier(df,col)

    print(df["Last_Rechenwert"].describe())

    df.to_csv("combinedData_cleared.csv")