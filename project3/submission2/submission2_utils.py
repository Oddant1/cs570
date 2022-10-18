import random
import torch

import pandas as pd


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
    returns the dataframe mutated by the mapping and the mapping
    """
    mapping = {}
    reverse_mapping = {}
    num_df = df.copy()

    for idx, column in enumerate(df.columns):
        if not pd.api.types.is_numeric_dtype(df[column]) or \
                column == 'MSSubClass':
            counter = 0
            mapping[idx] = {}
            reverse_mapping[idx] = {}

            for value in df[column]:
                if value in mapping[idx].keys():
                    continue

                mapping[idx][value] = counter
                reverse_mapping[idx][counter] = value
                counter += 1

            num_df = num_df.replace({column: mapping[idx]})
            reverse_mapping[idx]['MAX_VAL'] = counter

    return num_df, mapping, reverse_mapping


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


def create_houses(num_houses, df, mapping):
    for i in range(num_houses):
        house = []
        for column in df.loc[:, df.columns != 'SalePrice']:
            house.append(random.randint(df[column].min(), df[column].max()))
        yield house
