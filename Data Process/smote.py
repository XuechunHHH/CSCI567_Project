import pandas as pd 
from imblearn.over_sampling import SMOTE

train = pd.read_csv('train.csv')
y = train['Status']
X = train.drop(['Status'], axis = 1)

sm = SMOTE(random_state=42, sampling_strategy="minority")
X_res, y_res = sm.fit_resample(X, y)
print(X_res.shape)
print(y_res.shape)

train = pd.concat([X_res, y_res], axis=1)
print(train.shape)

train.to_csv('train_smote.csv', index=False)