#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# In[39]:


data = pd.read_csv('Clean.csv')
data.head()


# In[47]:


data.drop('Unnamed: 0',axis=1,inplace=True)


# In[48]:


df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])


# In[49]:


df.shape


# # Content based Recommendation system

# In[50]:


from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
scaler = MinMaxScaler()
normalization = df.select_dtypes(include=datatypes)
normalization = scaler.fit_transform(normalization)


# In[51]:


TWSS = []
k = list(range(1, 15))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    TWSS.append(kmeans.inertia_)


# In[52]:


plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[53]:


kmeans = KMeans(n_clusters=5)
features = kmeans.fit_predict(normalization)
data['features'] = features


# In[54]:


data[data['features']==0]


# In[55]:


data[data['features']==1]


# In[56]:


data[data['features']==2]


# In[57]:


data[data['features']==3]


# In[58]:


class Spotify_Recommendation():
    def __init__(self, dataset):
        self.dataset = dataset
    def recommend(self, songs, amount=1):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0
            for col in np.arange(len(rec.columns)):
                if not col in [1, 6, 12, 14, 18]:
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]


# In[67]:


recommendations = Spotify_Recommendation(data)
recommendations.recommend("Country Junction", 10)


# #  Popularity Based Recommedation System 

# In[62]:


data.head()


# In[66]:


data1 = data.groupby('name').mean()['popularity'].sort_values(ascending=False).head(10)


# In[64]:


data1


# In[ ]:




