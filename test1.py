import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## 讀取數據
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# change male and felmale  to 0 and 1
train_data['Sex'] = train_data['Sex'].replace({'male': 1, 'female': 0})
test_data['Sex'] = test_data['Sex'].replace({'male': 1, 'female': 0})

train_data['Has_Cabin'] = train_data['Cabin'].notna().astype(int)
test_data['Has_Cabin'] = test_data['Cabin'].notna().astype(int)

# 對沒有船艙號碼的資料，賦予一個代表數字，這裡用 -1 表示
train_data.loc[train_data['Has_Cabin'] == 0, 'Cabin_Code'] = -1
test_data.loc[train_data['Has_Cabin'] == 0, 'Cabin_Code'] = -1

# 對有船艙號碼的資料，提取字母部分並賦予編號，編號從 1 開始
mask = (train_data['Has_Cabin'] == 1) & (train_data['Cabin'].apply(lambda x: isinstance(x, str)))
train_data.loc[mask, 'Cabin_Code'] = train_data.loc[mask, 'Cabin'].str[0]
train_data.loc[mask, 'Cabin_Code'] = pd.Categorical(train_data.loc[mask, 'Cabin_Code']).codes + 1

mask = (test_data['Has_Cabin'] == 1) & (test_data['Cabin'].apply(lambda x: isinstance(x, str)))
test_data.loc[mask, 'Cabin_Code'] = test_data.loc[mask, 'Cabin'].str[0]
test_data.loc[mask, 'Cabin_Code'] = pd.Categorical(test_data.loc[mask, 'Cabin_Code']).codes + 1

# 將 'Cabin_Code' 列中的字符串轉換為數字
train_data['Cabin_Code'] = pd.to_numeric(train_data['Cabin_Code'], errors='coerce')

##選取六個數值當x值
test_data = test_data.dropna(subset=['Pclass', 'Sex', 'Age','Fare','Embarked','Has_Cabin'])
train_data = train_data.dropna(subset=['Pclass', 'Sex', 'Age','Fare','Embarked','Has_Cabin'])
#印出DataFrame
print(train_data)

X = train_data[['Pclass', 'Sex', 'Age','Fare','Embarked','Has_Cabin']]
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train, columns=['Embarked'])
X_val = pd.get_dummies(X_val, columns=['Embarked'])

##訓練模型
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=None)
model.fit(X_train, y_train)

## 預測驗證集
predictions = model.predict(X_val)

## 評估模型
accuracy = accuracy_score(y_val, predictions)
print(f'验证集准确度：{accuracy:.2f}')

## 在测试数据上进行预测
X_test = test_data[['Pclass', 'Sex', 'Age','Fare','Embarked','Has_Cabin']]

#對X_test進行one-hot encoding
X_test = pd.get_dummies(X_test, columns=['Embarked'])

# 確認測試集特徵和訓練集特徵一致
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

# 確認特徵順序一致
X_test = X_test[X_train.columns]

test_predictions = model.predict(X_test)

## 生成提交文件
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
submission.to_csv('titanic_submission.csv', index=False)
