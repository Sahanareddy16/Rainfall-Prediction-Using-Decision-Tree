#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification


# ##### filepath=(r"C:\Users\sahan\Finaldataset.csv")
# df=pd.read_csv(filepath)
# df

# In[4]:


df.isnull().sum()


# In[5]:


df =df.fillna(0)


# In[6]:


df.isnull().sum()


# In[7]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['District'] = label_encoder.fit_transform(df['District'])
df['Mandal'] = label_encoder.fit_transform(df['Mandal'])
df['Date'] = label_encoder.fit_transform(df['Date'])


# In[8]:


df


# In[9]:


df.info()


# In[10]:


df.columns


# In[11]:


df.shape


# In[12]:


features = list(df.select_dtypes(include = np.number).columns)
print(features)


# In[41]:


df.dtypes


# In[42]:


df.describe().T


# In[13]:


df.nunique()


# In[14]:


df.head


# In[15]:


df.tail


# In[44]:


df.corr()


# In[45]:


corr = df.corr()
plt.subplots(figsize=(25,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[46]:


sns.heatmap(df.isnull(),yticklabels=False, cmap="viridis")


# In[47]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# In[48]:


subData=df[['District', 'Mandal', 'Date', 'Temp', 'Humidity', 'Wind speed',
       'Rainfall' ]]
sns.pairplot(subData)


# In[49]:


plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.boxplot(df[col])
plt.tight_layout()
plt.show()


# In[50]:


from math import log
eps=np.finfo(float).eps
def ent(df,attribute):
    target_variables=df.Rainfall.unique()
    variables=df[attribute].unique()
    
    entropy_attribute=0
    for variable in variables:
        entropy_each_feature=0
        for target_variable in target_variables:
            num=len(df[attribute][df[attribute]==variable][df.Rainfall==target_variable])
            den=len(df[attribute][df[attribute]==variable])
            fraction=num/(den+eps)
            entropy_each_feature+=-fraction*log(fraction+eps)
        fraction2=den/len(df)
        entropy_attribute+=-fraction2*entropy_each_feature
    
    return(abs(entropy_attribute))


# In[51]:


a_entropy={k:ent(df,k) for k in df.keys()[:-1]}
a_entropy


# In[57]:



X, t = make_classification(100, 5, n_classes=2, shuffle=True, random_state=10)
X_train, X_test, t_train, t_test = train_test_split(
    X, t, test_size=0.26, shuffle=True, random_state=1)
 
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, t_train)
 
predicted_value = model.predict(X_test)
print(predicted_value)
 
tree.plot_tree(model)


# In[58]:


zeroes = 0
ones = 0
for i in range(0, len(t_train)):
    if t_train[i] == 0:
        zeroes += 1
    else:
        ones += 1
 
print(zeroes)
print(ones)
 
val = 1 - ((zeroes/70)*(zeroes/70) + (ones/70)*(ones/70))
print("Gini :", val)
 


# In[59]:


match = 0
UnMatch = 0
 
for i in range(25):
    if predicted_value[i] == t_test[i]:
        match += 1
    else:
        UnMatch += 1
 
accuracy = match/25
print("Accuracy is: ", accuracy)


# In[60]:


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.7,size = 500)
predicted = numpy.random.binomial(1,.7,size = 500)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


# In[ ]:




