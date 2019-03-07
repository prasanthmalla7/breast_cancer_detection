import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import neighbors
import pandas as pd

#pd.options.display.max_columns = 100
df = pd.read_csv('D:\py_scripts\k_nearest\Breast_cancer_wisconsin.txt')
df.replace('?',-99999,inplace = True)
df.drop(['id'],1,inplace = True)
#print(df.head(100))
x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

# splitting our data into train and test cases
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#using 
classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)

accuracy = classifier.score(x_test,y_test)
print(accuracy)

example = np.array([[4,2,1,1,3,2,1,2,1]])
example = example.reshape(len(example),-1)

prediction = classifier.predict(example)

print(prediction)