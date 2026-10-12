#Veri seti incelemesi
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target_label"]=cancer.target

#Modelin train edilmesi
X=cancer.data
y=cancer.target

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

#Normalization
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)


#Training
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

#Prediction
y_pred = knn.predict(X_test)

#Accuracy ve confusion matrix bulma
acc=accuracy_score(y_test, y_pred)
print("Accuracy: ",acc)

confu_matrix= confusion_matrix(y_test, y_pred)
print("Confusion matrix: ",confu_matrix)


#Hiperparametre ayarlaması
acc_values=[]
k_values=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    acc_values.append(acc)
    k_values.append(k)


plt.figure()
plt.plot(k_values, acc_values, marker= "o", linestyle= "-")
plt.title("K hiperparametresine göre accuracy değerleri")
plt.xlabel("K değerleri")
plt.ylabel("Accuracy değerleri")
plt.xticks(k_values)
plt.grid(True)













