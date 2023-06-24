#### import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import sklearn as skl
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#### import training and testing dataset
train_df = pd.read_csv("C:/Users/apple/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/apple/Downloads/test.csv")

#### Transform training data set
train_df['Female'] = \
    np.where(train_df['Sex'] == 'female','1',
    np.where(train_df['Sex'] == 'male','0','Unknown'
    )
)

train_df_sub = train_df[['Survived','PassengerId','Pclass','Female','Fare']]
#train_df_sub = train_df[['Survived','PassengerId','Pclass','Female','SibSp','Parch','Fare']]

#### Transform testing data set
test_df['Female'] = \
    np.where(test_df['Sex'] == 'female','1',
    np.where(test_df['Sex'] == 'male','0','Unknown'
    )
)

test_df_sub = test_df[['PassengerId','Pclass','Female','Fare']]
#test_df_sub = test_df[['PassengerId','Pclass','Female','SibSp','Parch','Fare']]

#### create a correlation graph of the different fields against the survival field
corr = train_df_sub.corr()
sns.heatmap(corr, cmap = 'RdBu', annot = True, xticklabels = corr.columns.values,yticklabels = corr.columns.values)
plt.show()

# We see a few fields that are correlated with Survival:
# Female is positively correlated with Survival at .54. So if a person is female they are moderately positive correlated with surviving. Basically the women had a positive correlation for surviving.
# We see a negatively correlation of -.34 between PClass and survival. This means that we have a weak negative correlation between Pclass going higher (higher means lower class)
# and the Survival rate. Basically saying that typically the higher class people survived.
# We also saw a weak positive correlation between Fare and Survival. As Fare increased, so did passenger class and so did survival.


#### attempt a logistic regression model on the training set as a first submission
y_train = train_df_sub['Survived']
x_train = train_df_sub[['Pclass','Female','Fare']]
#x_train = train_df_sub[['Pclass','Female','SibSp','Parch','Fare']]
#x = train_df_sub.drop('Survived', axis = 1) # Another way to get x values

model = LogisticRegression()  # create object for the class
model = model.fit(x_train,y_train)
model.intercept_
model.coef_

df_test = test_df_sub.set_index(['PassengerId'])
df_test['Survived'] = model.predict(df_test.fillna(0))
df_test[['Survived']].to_csv('LogisticRegressionOutput.csv',index=True)





#### figure out how to use xgboost and use that model as a second submission (Watch a youtube video on how to setup the model with training data)
modelxgb = xgb.XGBClassifier(tree_method = "gpu_hist",enable_categorical=True)

x_train['Female'].astype(str).astype(int) # Converts object type to int

modelxgb = modelxgb.fit(x_train,y_train)
print(modelxgb)

df_test = test_df_sub.set_index(['PassengerId'])
df_test['Female'].astype(str).astype(int) # Converts object type to int
df_test.dtypes
df_test['Survived'] = modelxgb.predict(df_test.fillna(0))
df_test[['Survived']].to_csv('XGBoostOutput.csv',index=True)