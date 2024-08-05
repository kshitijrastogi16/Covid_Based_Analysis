#!/usr/bin/env python
# coding: utf-8

# # Mental Health of Pregnant Women Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Pregnancy_Data.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df['Edinburgh_Postnatal_Depression_Scale'].fillna(value = round(df['Edinburgh_Postnatal_Depression_Scale'].mean()), inplace = True)


# In[6]:


df.isnull().sum()


# In[7]:


df['PROMIS_Anxiety'].fillna(value = round(df['PROMIS_Anxiety'].mean()), inplace = True)


# In[8]:


df.isnull().sum()


# In[9]:


df['Gestational_Age_At_Birth'].fillna(value = df['Gestational_Age_At_Birth'].mean(), inplace = True)


# In[10]:


df.isnull().sum()


# In[11]:


df['Birth_Length'].fillna(value = df['Birth_Length'].mean(), inplace = True)


# In[12]:


df.isnull().sum()


# In[13]:


df['Birth_Weight'].fillna(value = df['Birth_Weight'].mean(), inplace = True)


# In[14]:


df.isnull().sum()


# In[15]:


col = ['Delivery_Date(converted to month and year)', 'Delivery_Mode', 'NICU_Stay']
for i in col:
    df[i].fillna(method = 'ffill', inplace = True)


# In[16]:


df.isnull().sum()


# In[17]:


col = ['Delivery_Date(converted to month and year)', 'Delivery_Mode', 'NICU_Stay']
for i in col:
    df[i].fillna(method = 'bfill', inplace = True)


# In[18]:


df.isnull().sum()


# In[19]:


df.drop(columns=['OSF_ID', 'Maternal_Education', 'Delivery_Date(converted to month and year)', 'Language'], inplace=True)


# In[20]:


df.head()


# In[21]:


replace_map1 = { 'Caesarean-section (c-section)': 1, 'Vaginally': 0, 'Caesarean-section (c-section)': 1}
df['Delivery_Mode'] = df['Delivery_Mode'].replace(replace_map1)


# In[22]:


replace_map2 = {'Yes': 1, 'No': 0}
df['NICU_Stay'] = df['NICU_Stay'].replace(replace_map2)


# In[23]:


def encod(val):
    if val == 0:
        return 0
    else:
        return 1


# In[24]:


cols = ('Threaten_Life', 'Threaten_Baby_Danger', 'Threaten_Baby_Harm')
for i in cols:
    df[i] = df[i].apply(lambda x: encod(x))


# In[25]:


X = df.drop(columns = ['Threaten_Life', 'Threaten_Baby_Danger', 'Threaten_Baby_Harm'], axis = 1)
X.head()


# In[26]:


y_life_threat = df['Threaten_Life']
y_baby_danger = df['Threaten_Baby_Danger']
y_baby_harm = df['Threaten_Baby_Harm']


# In[27]:


y_life_threat.head()
y_baby_danger.head()
y_baby_harm.head()


# # Threaten_life as Target

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_life_threat_train, y_life_threat_test = train_test_split(X, y_life_threat, test_size=0.12, random_state=42)


# In[ ]:





# In[29]:


from sklearn.linear_model import LinearRegression
model_tl = LinearRegression()
model_tl.fit(X_train, y_life_threat_train)


# In[30]:


pred_tl = model_tl.predict(X_test)
binary_pred_tl = (pred_tl >= 0.5).astype(int)


# In[31]:


from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
accuracy_life_threat = accuracy_score(y_life_threat_test, binary_pred_tl)
mse_life_threat = mean_squared_error(y_life_threat_test, pred_tl)
r2_life_threat = r2_score(y_life_threat_test, pred_tl)


# In[32]:


print("Accuracy is:", round(accuracy_life_threat * 100, 2), "%")
print("Mean Squared Error is:", mse_life_threat)
print("R^2 Score is:", r2_life_threat)


# # Threaten_Baby_Danger as Target

# In[ ]:





# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_baby_danger_train, y_baby_danger_test = train_test_split(X, y_baby_danger, test_size=0.37, random_state=42)


# In[34]:


from sklearn.linear_model import LinearRegression
model_tbd = LinearRegression()
model_tbd.fit(X_train, y_baby_danger_train)


# In[35]:


pred_tbd = model_tbd.predict(X_test)
binary_pred_tbd = (pred_tbd >= 0.5).astype(int)


# In[36]:


from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
accuracy_baby_danger = accuracy_score(y_baby_danger_test, binary_pred_tbd)
mse_baby_danger = mean_squared_error(y_baby_danger_test, pred_tbd)
r2_baby_danger = r2_score(y_baby_danger_test, pred_tbd)


# In[37]:


print("Accuracy is:", round(accuracy_baby_danger * 100, 2), "%")
print("Mean Squared Error is:", mse_baby_danger)
print("R^2 Score is:", r2_baby_danger)


# # Threaten_Baby_Harm as Target

# In[46]:





# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_baby_harm_train, y_baby_harm_test = train_test_split(X, y_baby_harm, test_size=0.19, random_state=42)


# In[39]:


from sklearn.linear_model import LinearRegression
model_tbh = LinearRegression()
model_tbh.fit(X_train, y_baby_harm_train)


# In[40]:


pred_tbh = model_tbh.predict(X_test)
binary_pred_tbh = (pred_tbh >= 0.5).astype(int)


# In[41]:


from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
accuracy_baby_harm = accuracy_score(y_baby_harm_test, binary_pred_tbh)
mse_baby_harm = mean_squared_error(y_baby_harm_test, pred_tbh)
r2_baby_harm = r2_score(y_baby_harm_test, pred_tbh)


# In[42]:


print("Accuracy is:", round(accuracy_baby_harm * 100, 2), "%")
print("Mean Squared Error is:", mse_baby_harm)
print("R^2 Score is:", r2_baby_harm)

