import pandas as pd
df = pd.read_csv('iris.data')
df.tail()

import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X= df.iloc[0:100, [0,2]].values
plt.figure(1)
plt.subplot(311)
plt.scatter(X[:50,0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal_length')
plt.ylabel('sepal_length')
plt.legend(loc='upper left')




from Perceptron import Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.subplot(312)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassificationx')

from util import Util as utl
plt.subplot(313)
utl.plot_decision_regions(X, y, ppn, np, plt)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()