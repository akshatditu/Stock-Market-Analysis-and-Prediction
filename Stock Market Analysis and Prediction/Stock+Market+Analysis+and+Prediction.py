
# coding: utf-8

# # PROJECT : Stock Market Analysis and Prediction using ML

# ## Introduction
# 
#     Stock Market Analysis and Prediction is the project on technical analysis, visualization and prediction using data of few S&P 500 Companies.
#    

# ### Libraries Used: 
#     Numpy,Pandas,Matplotlib,Seaborn,Scikit Learn 
#    

# ### Actions Performed: 
#     Using the above python libraries, we have performed technical analysis i.e graphically presented moving averages of stocks, calculated Daily Returns and plotted them on graphs using visualization libraries. Later, we have predicted the future stock prices using three different models-Linear Regression, Decision Tree and Support Vector regression and also calculated the accuracy using mean-sqaured method.
# 

# ### About the Dataset :
#     It is the stock market data of Apple Inc. over the time period 04-01-2016 to 07-02-2018. Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services.
#     The dataset contains following columns-:
#         Date -> The particaular date on which the data is collected
#         Open -> The opening price of the stock on that day
#         High -> The highest price of the stock on that day
#         Low  -> The lowest price of the stock on that day
#         Close-> The closing price of the stock on that day
#         Volume->The volume of stock traded on that day

# ### Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (16, 7)


# ### Reading Dataset

# In[3]:


df = pd.read_csv('AAPL_data.csv')
df['Date']=pd.to_datetime(df['Date'])


# In[4]:


# Data Values in the dataset

print(df.head())
print('----')
print(df.tail())


# In[12]:


# Dataset description i.e mean,standard deviation,count,maximum value etc

df.describe()


# In[6]:


df.info()


# In[7]:


# Graphical representation of the closing price of the stock over time

df['Close'].plot(legend=True, figsize=(16,7))
plt.xlabel('Trading Days')
plt.ylabel('Price USD')
plt.title("Apple Stocks(04-01-2016 to 07-02-2018)")
plt.show()


# In[8]:


# Graphical representation of the Volume of the stock traded over time


df['Volume'].plot(legend=True, figsize=(16,7))
plt.xlabel('Trading Days')
plt.ylabel('Volume')
plt.title("Apple Stocks(04-01-2016 to 07-02-2018)")
plt.show()


# # Moving Average of Stock
# 
# ### Defintion: 
#     The moving average (MA) is a simple technical analysis tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks or any time period the trader chooses.

# In[9]:


#Calculating Moving Average

ma_days=[10,20,50,100]

for ma in ma_days:
    Col_name="MA %s days"%(str(ma))
    df[Col_name]=pd.rolling_mean(df['Close'],ma)


# In[9]:


df[['Close','MA 10 days','MA 20 days','MA 50 days','MA 100 days']].plot(subplots=False,figsize=(17,6))


# # Daily Return
# 
# ### Definition:
#     It is the calculation of how much you gained or lost per day of stock, subtract the opening price from the closing price.
# 

# In[13]:


# We'll use pct_change to find the percent change for each day

df['Daily Return']= df['Close'].pct_change()

df['Daily Return'].plot(figsize=(16,7), legend=True, linestyle='--', marker='o')


# In[21]:


df['Daily Return'].hist(figsize=(14,7),bins=100)
plt.xlabel('Daily Return Values')
plt.ylabel('No.Of Days')
plt.title("Daily Return Histogram")
plt.show()


# In[27]:


sns.distplot(df['Daily Return'].dropna(),bins=100,color='green')


# In[28]:


AAPL = pd.read_csv('AAPL_data.csv')
AMZN = pd.read_csv('AMZN_data.csv')
GOOGL = pd.read_csv('GOOGL_data.csv')
MSFT = pd.read_csv('MSFT_data.csv')


# In[29]:


d = {'AAPL':AAPL['Close'],'AMZN':AMZN['close'],'MSFT':MSFT['close'],'GOOGL':GOOGL['close']}
closingprices_df = pd.DataFrame(d) 


# In[30]:


closingprices_df.head(10)


# In[39]:


sns.jointplot(x='GOOGL',y='AMZN',data=closingprices_df,kind='hex',color='red')


# In[43]:


sns.jointplot(x='AAPL',y='MSFT',data=closingprices_df,kind='reg', color='red')


# In[44]:


comp_return = closingprices_df.pct_change()


# In[45]:


comp_return.head(10)


# In[53]:


sns.pairplot(data=comp_return.dropna().head(300),size=3,kind='scatter',diag_kind='kde')


# In[63]:


sns.jointplot(x='GOOGL',y='AMZN',data=comp_return.dropna(),kind='reg',color='orange',size=8)


# In[65]:


sns.jointplot(x='AAPL',y='MSFT',data=comp_return.dropna(),kind='kde', color='orange',size=8)


# ## Predicting Future Stock Prices
# 
#     Here, we will predict the closing prices of the stock using the open,high and low prices of the stock.

# ## Data Preprocessing
# 
#     Getting data ready for fitting into our models to get our prediction results

# In[60]:


df = pd.read_csv("AAPL.csv")


# In[61]:


#Dividing the dataset columns into X and y

X = df.iloc[:,1:4].values
y= df.iloc[:,4].values

y = y.reshape(-1,1)


# ## Standardisation
# 
#     The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1. Given the distribution of the data, each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.
#     
#     
# 
#     
# 
#     

# In[62]:


# Using Standard Scaler Library for Standardisation

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_X.fit_transform(y)


# ## Train Test Split
# 
#     Here, we split our data into training and testing sets. We train our model on the training set data and the predictions are further done on the test set. We use Scikit Learns train_test_split for the same.

# In[63]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)


# ## Prediction Models

# ### Linear Regression
# 
#     In statistics, linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).
#     

# In[64]:


from sklearn.linear_model import LinearRegression

regressor2 = LinearRegression()
regressor2.fit(X_train,y_train)
y_pred1 = regressor2.predict(X_test)
y_pred1 = sc_X.inverse_transform(y_pred1)
y_test1 = sc_X.inverse_transform(y_test)


# In[65]:


print(y_pred1[1])
print(y_test1[1])


# In[66]:


a= np.arange(0,378)    
plt.figure(figsize=(16,9))
plt.scatter(a,y_pred1,color='red')
plt.scatter(a,y_test1,color='blue')
plt.legend(labels=['y_pred','y_test'])
plt.ylabel('Price USD')
plt.xlabel('Trading days')
plt.title('Test Values VS Predicted Values-- Linear Regression')
plt.show()


# ### Support Vector Regression
# 
#     Support Vector Machine can also be used as a regression method, maintaining all the main features that characterize the algorithm (maximal margin). The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. 

# In[67]:


from sklearn.svm import SVR

regressor = SVR(kernel='rbf',epsilon=0.1)
regressor.fit(X_train,y_train)
y_pred2 = regressor.predict(X_test)
y_pred2 = sc_X.inverse_transform(y_pred2)


# In[68]:


print(y_pred2[1])
print(y_test1[1])


# In[72]:


plt.figure(figsize=(16,9))
plt.scatter(a,y_pred2,color='red')
plt.scatter(a,y_test1,color='green')
plt.legend(labels=['y_pred','y_test'])
plt.ylabel('Price USD')
plt.xlabel('Trading days')
plt.title('Test Values VS Predicted Values-- Support Vector Regression')
plt.show()


# ### Decision Tree Regression
#     
#     Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. 

# In[70]:


from sklearn.tree import DecisionTreeRegressor

regressor3 = DecisionTreeRegressor(random_state=0)
regressor3.fit(X_train,y_train)
y_pred3 = regressor3.predict(X_test)
y_pred3 = sc_X.inverse_transform(y_pred3)


# In[71]:


print(y_pred3[1])
print(y_test1[1])


# In[76]:


plt.figure(figsize=(16,9))
plt.scatter(a,y_pred3,color='red')
plt.scatter(a,y_test1,color='purple')
plt.legend(labels=['y_pred','y_test'])
plt.ylabel('Price USD')
plt.xlabel('Trading days')
plt.title('Test Values VS Predicted Values-- Support Vector Regression')
plt.show()


# ### Prediction Scores
# 
#     

# In[88]:


from sklearn.metrics import r2_score,mean_squared_error

print("Linear Regression :\t\t{}".format(r2_score(y_test1,y_pred1)))

print("Support Vector Regression: \t{}".format(r2_score(y_test1,y_pred2)))

print("Decision Tree Regression :  \t{}".format(r2_score(y_test1,y_pred3)))


# # Thank You 
