#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r"C:\Users\Rahul\OneDrive\Desktop\House_Rent_Dataset.csv")


# In[4]:


df


# In[ ]:





# In[41]:


df.head()


# In[42]:


df.tail()


# In[43]:


df.isnull().sum()


# In[44]:


df.info()


# In[45]:


df.duplicated().sum()


# In[46]:


df.describe()


# In[47]:


df.shape


# In[5]:


df.value_counts()


# In[48]:


df.columns


# In[49]:


print('Mean House Rent:',round(df['Rent'].mean()))
print('Max House Rent:',round(df['Rent'].max()))
print('Min House Rent:',round(df['Rent'].min()))


# # Highest House Rent 

# In[50]:


df['Rent'].sort_values(ascending=False)[:5]


# # Lowest House Rent

# In[51]:


df['Rent'].sort_values()[:5]


# # Barplot for rent in cities

# In[52]:


sns.set_context('poster',font_scale = 1)
plt.figure(figsize = (15,5))
M=df['City'].value_counts().plot(kind = "bar",color='green',rot=0)
for k in M.patches: 
    M.annotate(int(k.get_height()),(k.get_x() + .25 , k.get_height() -200),ha = 'center',color='black')
plt.show()


# # PIE CHART FOR RENT IN CITIES

# In[82]:


plt.figure(figsize= (20,7))
M1=df['City'].value_counts()
explode = [0,0,0,0,0.2,0]

M1.plot(kind='pie',explode = explode, colors = sns.color_palette(palette = 'bright'),autopct = "%.1f%%")

plt.legend(labels = M1.index, loc = 'best' )

plt.axis('equal')

plt.show()


# # SCATTER PLOT

# In[86]:


plt.ticklabel_format(style = 'plain')
plt.scatter(df['Size'],df['Rent'],color = 'b')
plt.xlabel('Size')
plt.ylabel('Rent')
plt.show()


# # HISTOGRAM

# In[92]:


plt.figure(figsize=(20,8))
df['Size'].hist(bins = 100,color = 'black')
plt.show()


# # HEATMAP ON BHK VS AREA TYPE
# 

# In[94]:


plt.figure(figsize=(20,5))
sns.heatmap(pd.crosstab(df['Area Type'],df['BHK']),cmap='Blues')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





#  

# # Outlier Analysis

# In[95]:


df1=df


# In[103]:


fig, axs = plt.subplots(2,3, figsize = (20,8))
plt1 = sns.boxplot(df['Rent'], ax = axs[0,1])
plt2 = sns.boxplot(df['Size'], ax = axs[0,2])
plt.tight_layout()


# In[4]:


#Outlier treatment for price

plt.boxplot(df.Rent)
Q1 = df.Rent.quantile(0.25)
Q3 = df.Rent.quantile(0.75)
IQR = Q3-Q1
housing = df[(df.Rent >= Q1 - 1.5*IQR) & (df.Rent <= Q3 + 1.5*IQR)]
housing.shape


# In[109]:


#Outlier treatment for price

plt.boxplot(df.BHK)
Q1 = df.BHK.quantile(0.25)
Q3 = df.BHK.quantile(0.75)
IQR = Q3-Q1
housing = df[(df.BHK >= Q1 - 1.5*IQR) & (df.BHK <= Q3 + 1.5*IQR)]
housing.shape


# In[24]:


#Outlier treatment for price

plt.boxplot(df.Size)
Q1 = df.Size.quantile(0.25)
Q3 = df.Size.quantile(0.75)
IQR = Q3-Q1
housing = df[(df.Size >= Q1 - 1.5*IQR) & (df.Size <= Q3 + 1.5*IQR)]


# # Feature Selection or Engineering

# ### DATA SPLITING

# In[3]:


#pip install category_encoders


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import category_encoders as ce


# In[8]:


df.info()


# In[9]:


#assuming x is your feature set y is your target
x= housing.drop('Rent',axis=1)
y=housing['Rent']


# In[10]:


encoder = ce.LeaveOneOutEncoder()
x= encoder.fit_transform(x,y)


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.20,random_state = 40)


# In[12]:


model_rfr = RandomForestRegressor (n_estimators = 7)
model_dt = DecisionTreeRegressor()
model_lr = LinearRegression()


# In[13]:


models = [model_rfr,model_dt,model_lr]


# In[14]:


for model in models :
    print(f"fitting model:{model}")
    model.fit(x_train,y_train)


# In[15]:


for model in models:
    print(f"score of {model} for training data : {model.score(x_train,y_train)}")


# In[16]:


for model in models:
    print(f"score of {model} for testing data : {model.score(x_test,y_test)}")


# In[17]:


for model in models [:]:
    y_predicted = model.predict


# In[27]:


def regression_results(y_true,y_pred):
    #regression metrics
    #explained_variance = metrics.explained_variance(y_true,y_pred)
    mean_absolute_error = metrics. mean_absolute_error(y_true,y_pred)
    mse=metrics.mean_squared_error(y_true,y_pred)
    #mean_squared_log_error = metrics.mean_squared_log_error(y_true,y_pred)
    #median_absolute_error = metrics.median_absolute_error(y_train,y_pred)
    r2=metrics.r2_score(y_true,y_pred)

    #print("explained_varience :",round(explained_variance,d))
    #print()
    print('r2:',round(r2,4))
    print('MAE:',round(mean_absolute_error,4))
    print('MSE:',round(mse,4))
    print('RMSE:',round(np.sqrt(mse),4))
    #print('median_absolute_error',round(median_abslolute_error,4))


# In[28]:


for model in models:
    y_pred = model.predict(x_test)
    
    print(f"Report:{model}")
    print(f"{regression_results(y_test,y_pred)}\n")


# In[ ]:





# In[ ]:




