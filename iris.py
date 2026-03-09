import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the famous Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("Dataset shape:", df.shape)
df.head()

# How many of each flower type?
print(df['species'].value_counts())


# Plot each species by petal size
colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}

plt.figure(figsize=(8, 5))
for species, color in colors.items():
    subset = df[df['species'] == species]
    plt.scatter(subset['petal length (cm)'], 
                subset['petal width (cm)'], 
                c=color, label=species, alpha=0.7)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Species by Petal Size')
plt.legend()
plt.show()



X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN Classifier - classifies by looking at nearest neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Model trained successfully!")



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


# Try predicting a brand new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]], 
    columns=iris.feature_names
)
prediction = model.predict(new_flower)
print(f"This flower is: {prediction[0]}")