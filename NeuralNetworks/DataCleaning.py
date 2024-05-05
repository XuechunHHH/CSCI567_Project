#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import pandas as pd


# In[5]:


# Read file
file_path = 'train.csv'
df = pd.read_csv(file_path)

# Print the shape
print(df.shape)


# In[7]:


# Drop any row that has at least one NA value
clean_df = df.dropna()

print(clean_df.shape)




# In[10]:


## only calculate mean for numeric columns
df_numeric = df.select_dtypes(include=[np.number])
df_non_numeric = df.select_dtypes(exclude=[np.number])

df_numeric_filled = df_numeric.fillna(df_numeric.mean())

df_non_numeric_dropped = df_non_numeric.dropna()

df_final = pd.concat([df_numeric_filled, df_non_numeric_dropped], axis=1)

df_final.dropna(inplace=True)

print(df_final.shape)

print(df_final)