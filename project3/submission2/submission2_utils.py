import torch
import random
from collections import namedtuple

import pandas as pd


class House():
    def __init__(self, fitness, secondary_fitness, fields):
        self.fitness = fitness
        self.secondary_fitness = secondary_fitness
        self.fields = fields

    def __repr__(self):
        return f'{self.fitness}, {self.secondary_fitness}: {self.fields}'


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(79, 178)  # 79-(61-61)-1
        self.hid2 = torch.nn.Linear(178, 178)
        self.hid3 = torch.nn.Linear(178, 178)
        self.oupt = torch.nn.Linear(178, 1)
        torch.nn.init.xavier_uniform_(self.hid1.weight)  # glorot
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.hid3.weight)
        torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = self.oupt(z)  # no activation, aka Identity()
        return z


def categorical_to_numeric(df):
    """Takes a dataframe and for every non-numeric column in the dataframe it
    maps each observed value in that column to a unique numerical value. It
    returns the dataframe mutated by the idx_mapping and the idx_mapping
    """
    idx_mapping = {}
    column_mapping = {}
    num_df = df.copy()

    for idx, column in enumerate(df.columns):
        idx_mapping[idx] = {}
        column_mapping[column] = idx
        if not pd.api.types.is_numeric_dtype(df[column]) or \
                column == 'MSSubClass':
            counter = 0

            for value in df[column]:
                if value in idx_mapping[idx].keys():
                    continue

                idx_mapping[idx][value] = counter
                counter += 1

            num_df = num_df.replace({column: idx_mapping[idx]})
            idx_mapping[idx]['MAX_VAL'] = counter

    return num_df, idx_mapping, column_mapping


def normalize(df):
    norm_df = df.copy()

    for column in df.columns:
        norm_df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    return norm_df


def normalize_single(df, values):
    norm_values = []
    for idx, column in enumerate(df.loc[:, df.columns != 'SalePrice']):
        norm_values.append((values[idx] - df[column].min()) / (df[column].max() - df[column].min()))

    return norm_values


def unnormalize(df, column, value):
    """Takes in a dataframe, a column, and a value then reverses the
    normalization on the value is if it were a value in the given column of the
    given dataframe
    """
    return value * (df[column].max() - df[column].min()) + df[column].min()


def create_houses(num_houses, df):
    for _ in range(num_houses):
        house = []
        for column in df.loc[:, df.columns != 'SalePrice']:
            house.append(random.randint(df[column].min(), df[column].max()))
        yield House(0, 0, house)
