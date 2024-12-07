import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

data = pd.read_csv('Tesla.csv')
data.head()

data.shape

data.describe()

data.info()

plt.figure(figsize=(15,5))
plt.plot(data['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

data = data.drop(['Adj Close'], axis=1)

data.isnull().sum()

splitted = data['Date'].str.split('/', expand=True)
data['day'] = splitted[1].astype('int')
data['month'] = splitted[0].astype('int')
data['year'] = splitted[2].astype('int')
data.head()

data['is_quarter_end'] = np.where(data['month']%3==0,1,0)
data.head()

data_grouped = data.drop('Date', axis=1).groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()

data.drop('Date', axis=1).groupby('is_quarter_end').mean()

data['open-close'] = data['Open'] - data['Close']
data['low-high'] = data['Low'] - data['High']
data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

x = data[['open-close', 'low-high', 'is_quarter_end']]
y = data['target']

scaler = StandardScaler()
x = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=2022)

model = LogisticRegression()
model.fit(xtrain, ytrain)
print('Training Accuracy : ', accuracy_score(ytrain, model.predict(xtrain)))
print('Validation Accuracy : ', accuracy_score(ytest, model.predict(xtest)))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, xtest, ytest)
plt.show()
