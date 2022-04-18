from datetime import date,timedelta, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image

# getting part of a date
df["week"]=df["date"].apply(lambda x: x.week)
df["date"] =  df["rtimestamp"].apply(lambda x: x.date())
df["year"] =  df["rtimestamp"].apply(lambda x: x.year)
df["month"] =  df["rtimestamp"].apply(lambda x: x.month)
df["day"] =  df["rtimestamp"].apply(lambda x: x.day)

# getting last hour
pd.to_datetime(datetime.now() - timedelta(hours = 1)).strftime('%Y/%m/%d/%H')
