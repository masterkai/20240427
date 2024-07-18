import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the dataset
df = pd.read_csv('https://raw.githubusercontent.com/ryanchung403/dataset/main/Housing_Dataset_Sample.csv')

# observe the first 10 rows of the dataset
df.head(n=10)
df.describe().T
sns.displot(df['Price'])
sns.jointplot(x='Avg. Area Income', y='Price', data=df)
sns.pairplot(df)

# prepare to train the model
X = df.iloc[:,:5]
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)

# train the model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# predict the test set
predictions = reg.predict(X_test)
# evaluate the model
from sklearn.metrics import r2_score
r2_score(y_test, predictions)
plt.scatter(y_test, predictions, alpha=0.1)