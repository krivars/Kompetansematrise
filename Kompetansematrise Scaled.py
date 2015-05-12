
# coding: utf-8

# In[1]:

import csv
import re
import numpy as np
import pandas
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from IPython.display import clear_output

get_ipython().magic(u'matplotlib inline')


# ### Parametere

# In[2]:

n_components = 2   # Kan ikke egentlig endre denne..


# ### Funksjoner

# In[3]:

def do_pca_scaled(data):
    
    data.dropna(how='any', inplace=True)
    
    rowsum = np.array(np.matrix(data)*np.matrix(np.ones(data.shape[1])).transpose())
    scaled = pandas.DataFrame(np.matrix(np.diag([1/float(a) for a in rowsum]))*np.matrix(data))
    scaled.columns = data.columns
    scaleddata = pandas.concat([data.reset_index()[['Navn','Team']],scaled], axis=1).set_index(['Navn','Team'])
    
    pca = PCA(n_components=n_components)
    pca.fit(scaleddata)
    transformed_data = pca.transform(scaleddata)
    newdata = pandas.concat([data.reset_index()[['Navn', 'Team']], pandas.DataFrame(transformed_data)], axis=1)
    
    components = pandas.concat([data.transpose().reset_index()['index'], pandas.DataFrame({'1':pca.components_[0]}), pandas.DataFrame({'2':pca.components_[1]})], axis=1).set_index('index')
    
    return newdata, components

def plot_pca(newdata):
    ms = 90
    ax = newdata[newdata['Team']=='IM'].plot(kind='scatter', x=0, y=1, s=ms, color='Turquoise', label='IM', figsize=(12,12))
    newdata[newdata['Team']=='BST' ].plot(kind='scatter', x=0, y=1, s=ms, color='Gold', label='BST', ax=ax)
    newdata[newdata['Team']=='ITST'].plot(kind='scatter', x=0, y=1, s=ms, color='Red', label='ITST', ax=ax)

    for navn, team, x, y in newdata.values:
        if team == 'IM':
            horz = 'right'
        else:
            horz = 'left'
        shortname = re.match('([A-Za-z]+)\,?', navn).group(1)
        plt.annotate(shortname, xy=(x, y), textcoords = 'offset points', xytext = (0,8), horizontalalignment = horz)
        
    pass

def plot_component(components, ind):
    
    plt.figure()
    components.sort(str(ind))[str(ind)].plot(kind='barh', figsize=(8,30), title='PCA component ' + str(ind));
    
    pass


# # Relativ kompetanse

# In[4]:

# Last data
data = pandas.DataFrame.from_csv('20150408 LBK Kompetanse_formatert.csv', sep=';', index_col=[0,1])

newdata, components = do_pca_scaled(data)

plot_pca(newdata)


# #### x-component

# In[18]:

plot_component(components, 1)


# #### y-component

# In[19]:

plot_component(components, 2)


# # Relativ interesse

# In[5]:

# Last data
data = pandas.DataFrame.from_csv('20150408 LBK Interesse_formatert.csv', sep=';', index_col=[0,1])

newdata, components = do_pca_scaled(data)

plot_pca(newdata)


# #### x-component

# In[21]:

plot_component(components, 1)


# #### y-component

# In[22]:

plot_component(components, 2)


# # Relativt kombinert?

# In[23]:

# Last data
data = pandas.DataFrame.from_csv('20150414 LBK Kombinert_formatert.csv', sep=';', index_col=[0,1])

newdata, components = do_pca_scaled(data)

plot_pca(newdata)


# #### x-component
# 

# In[24]:

plot_component(components, 1)


# #### y-component
# 

# In[25]:

plot_component(components, 2)


# In[ ]:

#col = np.array(data.iloc[0]) - np.array(pca.mean_)
#print pca.transform(np.array(data.iloc[0]))
#print sum( col * pca.components_[0] )
#print sum( col * pca.components_[1] )


# In[ ]:



