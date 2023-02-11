#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('spotify.csv')
df.head()


# # EXPLORATORY DATA ANALYSIS

# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


# examine duplicated rows
df.loc[df.duplicated(), :]


# In[8]:


data = df.drop_duplicates(keep="first")


# In[9]:


data


# In[10]:


df1 = df.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])
df1.corr()


# #  VISUALIZATION

# In[11]:


from matplotlib import cm
color = cm.inferno_r(np.linspace(.4, .8, 30))


# In[12]:


plt.figure(figsize=(10,4))
plt.title("Distribution of Popularity Variable.")
sns.distplot(data['popularity'],color='#8B1A1A');


# In[13]:


plt.figure(figsize=(10,3))
plt.title("Boxplot of Popularity Variable.")
sns.boxplot(data=data['popularity'],color='#8B1A1A',orient='horizontal');


# In[14]:


data.columns


# In[15]:


plt.figure(figsize=(10,4))
plt.title("Top 10 songs based on Popularity.")
data.groupby('name').mean()['popularity'].sort_values(ascending=False).head(10).plot(kind='bar',colormap='Paired');


# In[16]:


plt.figure(figsize=(10,4))
plt.title("Top 10 Artist based on Popularity.")
data.groupby('artists').mean()['popularity'].sort_values(ascending=False).head(10).plot(kind='bar',color=color,stacked=True)
plt.xlabel('Artists',fontsize=10)
plt.xticks(rotation=-70);


# In[17]:


plt.figure(figsize=(12,4))
plt.suptitle("Duration_ms Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['duration_ms'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['duration_ms'],color='#8B1A1A');


# In[18]:


plt.figure(figsize=(12,4))
plt.suptitle("danceability Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['danceability'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['danceability'],color='#8B1A1A');


# In[19]:


plt.figure(figsize=(12,4))
plt.suptitle("energy Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(df['energy'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(df['energy'],color='#8B1A1A');


# In[20]:


plt.figure(figsize=(12,4))
plt.suptitle("key Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['key'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['key'],color='#8B1A1A');


# In[21]:


plt.figure(figsize=(12,4))
plt.suptitle("loudness Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['loudness'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['loudness'],color='#8B1A1A');


# In[22]:


plt.figure(figsize=(12,4))
plt.suptitle("speechiness Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['speechiness'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['speechiness'],color='#8B1A1A');


# In[23]:


plt.figure(figsize=(12,4))
plt.suptitle("acousticness Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['acousticness'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['acousticness'],color='#8B1A1A');


# In[24]:


plt.figure(figsize=(12,4))
plt.suptitle("instrumentalness Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['instrumentalness'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['instrumentalness'],color='#8B1A1A');


# In[25]:


plt.figure(figsize=(12,4))
plt.suptitle("liveness Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['liveness'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['liveness'],color='#8B1A1A');


# In[26]:


plt.figure(figsize=(12,4))
plt.suptitle("valence Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['valence'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['valence'],color='#8B1A1A');


# In[27]:


plt.figure(figsize=(12,4))
plt.suptitle("tempo Varialble Statisitics.",fontsize=16)
plt.subplot(1,2,1)
sns.distplot(data['tempo'],color='#8B1A1A')
plt.subplot(1,2,2)
sns.boxplot(data['tempo'],color='#8B1A1A');


# In[28]:


plt.figure(figsize=(15,12))
sns.heatmap(df1.corr(),annot=True,cmap='Blues')
plt.show()

