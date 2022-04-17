import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")

# to widen the printed array
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}')) 

# see all rows and columns
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

# PLOTTING IMAGES

# to make plotting inline
# %matplotlib inline

# reading files with parsing
df = pd.read_csv("filepath.csv",parse_dates=["date_column"])

# one-hot encoding
def one_hot_encoder(df,col):
    one_hot= pd.get_dummies(df[col])
    df=df.drop(col,axis=1)
    new_df = pd.concat([df,one_hot],axis=1)
    return new_df

# label encoder
def label_encoder(df,col):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df

def split_data(data,label, test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.drop([label],axis=1), data[label], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test 

# read sql query
def create_pandas_table(sql_query, database = conn):
    table = pd.read_sql_query(sql_query, database)
    return table

# connect to postgre database
conn = psycopg2.connect(
       host="host",
       database="db",
       port ="port", 
       user="user",
       password="pswrd")

cursor = conn.cursor()

