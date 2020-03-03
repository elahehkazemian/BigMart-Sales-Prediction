#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[4]:



train = pd.read_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\Train.csv") 
test = pd.read_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\Test.csv") 


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.info()
train.describe()


# In[8]:


idsUq = len(set(train.Item_Identifier))
idsTot = train.shape[0]
idsDupli = idsTot - idsUq
print("Total " + str(idsDupli) + " duplicate found in " + str(idsTot) + " total entries")


# In[9]:


print ("Skew is:", train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())


# In[10]:


numfeats = train.select_dtypes(include=[np.number])
numfeats.dtypes


# In[11]:


numfeats.corr()


# In[12]:


corr = numfeats.corr()
print (corr['Item_Outlet_Sales'].sort_values(ascending=False))


# In[13]:


train.Item_Fat_Content.value_counts()


# In[14]:


train.Item_Type.value_counts()


# In[15]:


train.Outlet_Size.value_counts()


# In[16]:


train.Outlet_Location_Type.value_counts()


# In[17]:


train.Outlet_Type.value_counts()


# In[18]:


numfeat = train.select_dtypes(include=[np.object])
numfeat.dtypes


# In[19]:


train.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[20]:


train.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())


# In[21]:


train.pivot_table(values='Outlet_Location_Type', columns='Outlet_Type',aggfunc=lambda x:x.mode())


# In[22]:


train.pivot_table(index='Item_Type', values="Item_Visibility", aggfunc=np.mean)


# In[24]:


train['source']='train'
test['source']='test'

data = pd.concat([train,test], ignore_index = True)
data.to_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\data.csv",index=False)
print(train.shape, test.shape, data.shape)


# In[25]:


itemavgwt = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(itemavgwt)


# In[26]:


def impute_weight(cols):
    Weight = cols[0]
    Identifier = cols[1]
    
    if pd.isnull(Weight):
        return itemavgwt['Item_Weight'][itemavgwt.index == Identifier]
    else:
        return Weight


# In[27]:


print ('Orignal #missing: %d'%sum(data['Item_Weight'].isnull()))
data['Item_Weight'] = data[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1).astype(float)
print ('Final #missing: %d'%sum(data['Item_Weight'].isnull()))


# In[28]:


from scipy.stats import mode
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode


# In[29]:


def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size

print ('Orignal #missing: %d'%sum(data['Outlet_Size'].isnull()))
data['Outlet_Size'] = data[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull()))


# In[30]:


data.pivot_table(values='Item_Outlet_Sales', columns='Outlet_Type')


# In[31]:


visitemavg = data.pivot_table(values='Item_Visibility',index='Item_Identifier')


# In[32]:


def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visitemavg['Item_Visibility'][visitemavg.index == item]
    else:
        return visibility


# In[33]:



print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'] = data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))


# In[34]:


data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[35]:


data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[36]:


print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})

print(data['Item_Fat_Content'].value_counts())


# In[37]:


data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[38]:


func = lambda x: x['Item_Visibility']/visitemavg['Item_Visibility'][visitemavg.index == x['Item_Identifier']][0]
data['Item_Visibility_MeanRatio'] = data.apply(func,axis=1).astype(float)
data['Item_Visibility_MeanRatio'].describe()


# In[39]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[40]:


data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[41]:


data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])

data.dtypes


# In[56]:


data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
train.to_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\train_hsn.csv",index=False)
test.to_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\test_hsn.csv",index=False)


# In[ ]:


train_df = pd.read_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\train_hsn.csv")


# In[57]:


test_df = pd.read_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\IP\\test_hsn.csv")


# In[58]:


target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    alg.fit(dtrain[predictors], dtrain[target])
    dtrain_predictions = alg.predict(dtrain[predictors])
    cv_score = model_selection.cross_val_score(alg, dtrain[predictors],(dtrain[target]) , cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((dtrain[target]).values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    dtest[target] = alg.predict(dtest[predictors])
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\OP\\" + filename, index=False)


# In[59]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression(normalize=True)

predictors = train_df.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
modelfit(LR, train_df, test_df, predictors, target, IDcol, 'LR.csv')


# In[60]:


from sklearn.linear_model import Ridge
RR = Ridge(alpha=0.05,normalize=True)
modelfit(RR, train_df, test_df, predictors, target, IDcol, 'RR.csv')


# In[61]:


from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(DT, train_df, test_df, predictors, target, IDcol, 'DT.csv')


# In[ ]:





# In[62]:


from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
             eval_set=[(test_df[predictors], test_df[target])], verbose=False)


# In[63]:


train_df_predictions = my_model.predict(train_df[predictors])


# In[64]:


predictions = my_model.predict(test_df[predictors])


# In[65]:


from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))


# In[66]:


IDcol.append(target)
submission = pd.DataFrame({ x: test_df[x] for x in IDcol})
submission.to_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Bigmart sales\\OP\\hsn.csv", index=False)


# In[ ]:




