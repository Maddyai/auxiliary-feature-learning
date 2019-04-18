import numpy as np
np.random.seed(21337)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from helper import vcat_dist, vnum_dist
import pandas as pd


def preprocess_dataR2(df):
    x = df.drop('Classification', axis=1)
    # y = df['Classification']
    # one hot encoding
    y = pd.get_dummies(df['Classification'], drop_first=True)
    # y.columns = ["target"]
    return x, y


def preprocess_wine(df):
    df.coloums = [str(i) for i in range(df.shape[1])]
    x = df.drop("1", axis=1)
    y = pd.get_dummies(df['1'], drop_first=False)
    return x, y


def preprocess_audit(df):
    df = df.dropna()
    df = df[~df.LOCATION_ID.str.contains("LOHARU")]
    df = df[~df.LOCATION_ID.str.contains("NUH")]
    df = df[~df.LOCATION_ID.str.contains("SAFIDON")]
    x = df.drop("Risk", axis=1)
    y = pd.get_dummies(df['Risk'], drop_first=False)
    return x, y


class DataLoader():
    def __init__(self, dataset_name, dataset_path, preprocess):
        self.name = dataset_name
        self.df = pd.read_csv(dataset_path, delimiter=',')
        self.scaler = MinMaxScaler()
        self.preproc_func = preprocess

    def reset_scalar(self):
        self.scaler = MinMaxScaler()

    def split_dataset(self, training_size):
        df_train, self.df_test = train_test_split(self.df, test_size=1-training_size)
        self.df_train, self.df_val = train_test_split(df_train, test_size=0.25)

    def preprocess(self):
        self.x_train, self.y_train = self.preproc_func(self.df_train)
        self.x_val, self.y_val = self.preproc_func(self.df_val)
        self.x_test, self.y_test = self.preproc_func(self.df_test)

        self.x_train_scaled = self._scale_df(self.x_train, fit=True)
        self.x_val_scaled = self._scale_df(self.x_val)
        self.x_test_scaled = self._scale_df(self.x_test)

    def _scale_df(self, x, fit=False):
        if fit:
            return pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns)
        else:
            return pd.DataFrame(self.scaler.transform(x), columns=x.columns)