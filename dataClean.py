"""
tools for data cleaning
"""
import os
import glob
import numpy as np
from collections import Counter
import pandas as pd


def read_dfs(path):
    """
    :param path: path
    :return: concatenated df
    """
    os.chdir(path)
    files = [x for x in glob.glob('*')]
    files = [x for x in files if x[-3:] == 'csv']
    dfs = [pd.read_csv(f, header=None) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    col = list(data.iloc[0, :])
    data.columns = col
    data = data[data[col[0]] != col[0]]
    data = data.reset_index(drop=True)
    return data


def trans_num(df, col):
    """
    transform columns to numeric type
    :param df: data frame
    :param col: string, column name / str[], list of column names
    :return: nothing
    """
    df[col] = df[col].apply(pd.to_numeric, args=('coerce',))


def trans_num2category(df, col):
    """
    transform columns to numeric type
    :param df: data frame
    :param col: string, column name
    :return: nothing
    """
    df[col] = df[col].apply(pd.to_numeric, args=('coerce',)).astype(str)


def check_counts(df, col):
    """
    :param df: data frame
    :param col: string, column name
    :return: counts, json object
    """
    print(set(df[col]))
    print(Counter(df[col]))
    return dict(Counter(df[col]))


def get_summary(df, cols):
    """
    :param df: data frame
    :param cols: list of column names that you want to get summary info
    :return: mean, var, min, Q10, Q20, Q30, Q40, ... , Q90, max, missing percentage
    """
    col = ['feature', 'mean', 'var', 'min'] + ['Q'+str(10*x) for x in range(1, 10)] + ['max', 'missing%']
    summary = pd.DataFrame(columns=col)
    for feature in cols:
        row = dict()
        row['feature'] = feature
        row['mean'] = np.nanmean(df[feature])
        row['var'] = np.nanvar(df[feature])
        row['min'] = np.nanmin(df[feature])
        for q in range(1, 10):
            variable = 'Q'+str(10*q)
            row[variable] = np.nanpercentile(df[feature], 10*q)
        row['max'] = np.nanmax(df[feature])
        row['missing%'] = round(df[feature].isnull().sum()/len(df), 2)
        summary = summary.append(row, ignore_index=True)
    return summary


def get_corr_pair(df, threshold=0.8):
    """
    :param df: data frame
    :param threshold:
    :return: a table consists of highly correlated pairs
    """
    cor = df.corr()
    columns = list(cor.columns)
    x1, x2, correlation = [], [], []
    for i in range(len(cor)):
        for j in range(i):
            if cor[columns[i]][j] > threshold:
                x1.append(columns[i])
                x2.append(columns[j])
                correlation.append(cor[columns[i]][j])
    result = pd.DataFrame(list(zip(x1, x2, correlation)))
    return result


def filter_data(df, target_values, col_name):
    """
    :param df: data frame
    :param target_values: a list of values you want to keep
    :param col_name: str, the col you wang to do the filtering process
    :return: a subset of df
    """
    ind = []
    for item in df[col_name]:
        if item in set(target_values):
            ind.append(True)
        else:
            ind.append(False)
    return df.iloc[ind, ]


class Rescale:
    """
    A class to do rescaling work
    """
    def __init__(self, df):
            self.maximum = dict()
            self.minimum = dict()
            self.mean = dict()
            self.std = dict()
            self.df = df

    def min_max_ada(self, col):
        """
        A method for training set
        :param col: str, column name
        :return: nothing
        """
        self.maximum[col] = self.df[col].max()
        self.minimum[col] = self.df[col].min()
        self.df[col] = (self.df[col] - self.minimum[col]) / (self.maximum[col] - self.minimum[col])

    def min_max(self, col):
        """
        A method for test set, need to set self.maximum and self.minimum first
        :param col: str, column name
        :return: nothing
        """
        self.df[col] = (self.df[col] - self.minimum[col]) / (self.maximum[col] - self.minimum[col])

    def normalize_ada(self, col):
        """
        A method for training
        :param col: str, column name
        :return: nothing
        """
        self.mean[col] = np.nanmean(self.df[col])
        self.std[col] = np.nanstd(self.df[col])
        self.df[col] = (self.df[col] - self.mean[col]) / self.std[col]

    def normalize(self, col):
        """
        A method for test set, need to set self.mean and self.std first
        :param col: str, column name
        :return: nothing
        """
        self.df[col] = (self.df[col] - self.mean[col]) / self.std[col]



