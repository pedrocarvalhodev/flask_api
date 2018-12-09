
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestClassifier


# In[97]:


df = pd.read_csv("https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv")


# In[98]:


df.head(2)


# In[99]:


df.drop("model", axis=1, inplace=True)


# In[100]:


#pd.get_dummies(df.mod
df.head(2)


# In[101]:


df.info()


# In[102]:


df.columns


# In[103]:


print(df.nunique())
print(df.groupby("cyl").mpg.count())


# In[104]:


y = "hp"
X = [x for x in df.columns if x != y]

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.30, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[105]:


reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
y_pred = reg.predict(X_test)
yh  = [x for x in zip(y_test, y_pred)]
print(yh)
rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_pred))
print(rootMeanSquaredError)


# ## Variable Importance

# In[106]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
df_X = df[X].copy()
df_X['randomVar'] = np.random.randint(1, 6, df_X.shape[0])
clf = clf.fit(df_X, df[y])
features = pd.DataFrame()
features['feature'] = df_X.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features = features.sort_values(by="importance", ascending=False).reset_index(drop=False)
features


# In[107]:


randomVarIndex = features[features.feature=="randomVar"].index.values[0]


# In[108]:


feat_positive = list(features[features.index < randomVarIndex].feature.values)
feat_positive


# In[109]:


reg = LinearRegression(fit_intercept=True).fit(X_train[feat_positive], y_train)
y_pred = reg.predict(X_test[feat_positive])
yh  = [x for x in zip(y_test, map(int, y_pred))]
print(yh)
rootMeanSquaredError = sqrt(mean_squared_error(y_test, y_pred))
print(rootMeanSquaredError)


# In[110]:


# Compare variable importance with predictive capacity of each var with intercept, Mean RMSE with train-test loop 

