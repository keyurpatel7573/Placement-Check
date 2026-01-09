import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions 
import pickle

df =  pd.read_csv("students_placement_data.csv")
df.info()
df= df.iloc[:,1:]
plt.scatter(df["IQ"],df["CGPA"],c=df["Placement"])
 
X = df.iloc[:,0:2]
Y= df.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

# scale the value 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train the modal
clf = LogisticRegression()
clf.fit(X_train,Y_train)

# scoring Evalute the modal 
Y_predict = clf.predict(X_test)
accuracy_score(Y_test,Y_predict)


plot_decision_regions(X_train,Y_train.values,clf=clf, legend=2)
 
pickle.dump(clf,open("placement_model.pkl","wb"))
pickle.dump(scaler,open("scaler.pkl","wb"))
