'''This is a rebuilt version of the pandas library for 42AI students'''

import math
import pandas as pd

def count(series):
    '''Returns the number of elements in the Series'''
    cnt = 0
    for elem in series:
        if not math.isnan(elem):
            cnt += 1
    return cnt

def min(series):
    '''Returns the minimum value of the Series'''
    min_val = series[0]
    for elem in series:
        if not math.isnan(elem):
            if elem < min_val:
                min_val = elem
    return min_val

def max(series):
    '''Returns the maximum value of the Series'''
    max_val = series[0]
    for elem in series:
        if not math.isnan(elem):
            if elem > max_val:
                max_val = elem
    return max_val

def mean(series):
    '''Returns the mean of the Series'''
    total = 0
    cnt = 0
    for elem in series:
        if not math.isnan(elem):
            total += elem
            cnt += 1

    if cnt == 0:
        return float('NaN')
    return total / cnt

def variance(series, ddof=1):
    '''Returns the variance of the Series'''
    series = series.dropna()
    if series.empty:
        return float('NaN')
    return ((series - mean(series))**2).sum() / (count(series) - ddof)

def std(series, ddof=1):
    '''Returns the standard deviation of the Series'''
    series = series.dropna()
    if series.empty:
        return float('NaN')
    return math.sqrt(variance(series, ddof))

def analyse_col(series, ddof=1, percentiles=[0,25, 0,50, 0,75]):
    '''Returns a Series with the statistical description of the Series'''
    if series.empty:
        print(f'Error: Empty column {series.name}')
        exit(-1)

    data = {
        'count': count(series),
        'mean': mean(series),
        'std': std(series, ddof),
        'min': min(series),
    }

    for p in percentiles:
        data[f"{round(p*100)}%"] = calcular_percentil(series, p)
    
    data['max'] = max(series)

    return pd.Series(data)

def describe(df, ddof=1, percentiles=[0.25, 0.50, 0.75]):
    '''Returns a DataFrame with the statistical description of the DataFrame'''

    results = {}

    if 0.5 not in percentiles:
        percentiles.append(0.5)

    for i, col_name in enumerate(df):
        results[col_name] = analyse_col(df[col_name], ddof, percentiles)
    return pd.DataFrame(results)

def get_numeric_columns(df):
    '''Returns all numerical "features"'''
    num_cols = df.select_dtypes('number')
    # Drop empty columns
    num_cols = num_cols.dropna(axis=1, how='all')
    # Exclude 'Index' column
    return num_cols.loc[:, num_cols.columns != 'Index']

def calcular_percentil(data, percentile=0.5):
    '''Returns the value of the percentile'''

    data_sorted = list(data.sort_values().dropna())

    n = len(data_sorted)
    if not n:
        return float('NaN')
    index = (percentile) * (n - 1)
   
    #print(index)
    if index.is_integer():
        # If the calculated index is an integer
        return data_sorted[int(index)]
    else:
        # Linear interpolation if the index is not an integer
        low_index = int(index)
        high_index = low_index + 1
        interpolation = index - low_index

        return (1 - interpolation) * data_sorted[low_index] + interpolation * data_sorted[high_index]

def normalizer(df):
    '''Returns the normalized DataFrame'''
    normalized_df = df.copy()
    for col in normalized_df.columns:
        if normalized_df[col].dtype != 'float64':
            continue
        # Get the minimum and maximum of the current column
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()

        # Apply normalization to the column
        normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    return normalized_df

def cov(df):
    '''Returns the correlation matrix'''
    df = df.dropna()
    df = get_numeric_columns(df)

    means = df.apply(lambda x: mean(x))

    # Initializing a DataFrame for the covariance matrix
    cov_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculating the covariance matrix
    for col1 in df.columns:
        for col2 in df.columns:
            cov_matrix.loc[col1, col2] = mean((df[col1] - means[col1]) * (df[col2] - means[col2]))
    return cov_matrix

def corr(df):
    '''Returns the covariance matrix'''
    df = df.dropna()
    df = get_numeric_columns(df)

    stds = df.apply(lambda x: std(x, ddof=0))

    # Initializing a DataFrame for the covariance matrix
    cov_matrix = cov(df)

    # Initializing a DataFrame for the correlation matrix
    corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculating the correlation matrix
    for col1 in df.columns:
        for col2 in df.columns:
            corr_matrix.loc[col1, col2] = cov_matrix.loc[col1, col2] / (stds[col1] * stds[col2])

    return corr_matrix

def MinMaxScaler(df):
    '''Returns the MinMax scaled DataFrame'''
    df = df.dropna()
    df = get_numeric_columns(df)

    min_values = df.apply(lambda x: min(x))
    max_values = df.apply(lambda x: max(x))

    # Initializing a DataFrame to store the scaled values
    scaled_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Scaling the values
    for col in df.columns:
        scaled_df[col] = (df[col] - min_values[col]) / (max_values[col] - min_values[col])

    return scaled_df

def StandardScaler(df):
    '''Returns the Standard scaled DataFrame'''
    df = df.dropna()
    df = get_numeric_columns(df)

    means = df.apply(lambda x: mean(x))
    stds = df.apply(lambda x: std(x, ddof=0))

    # Initializing a DataFrame to store the scaled values
    scaled_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Scaling the values
    for col in df.columns:
        scaled_df[col] = (df[col] - means[col]) / stds[col]

    return scaled_df

def gradient_descent_step(theta0, theta1, X, Y, learning_rate):
    '''Returns the updated theta0 and theta1 values'''
    m = len(X)
    theta0_gradient = sum(theta0 + theta1 * X - Y) / m
    theta1_gradient = sum((theta0 + theta1 * X - Y) * X) / m

    new_theta0 = theta0 - learning_rate * theta0_gradient
    new_theta1 = theta1 - learning_rate * theta1_gradient


    return new_theta0, new_theta1