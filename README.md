import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
 
# Load the iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
 
# Convert to a DataFrame for easier understanding 
df = pd.DataFrame(X, columns=iris.feature_names) 
df['species'] = y 
 
# Display the first few rows of the dataset 
print(df.head()) 
 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
 
# Initialize the Decision Tree Classifier 
clf = DecisionTreeClassifier() 
 
# Train the model 
clf.fit(X_train, y_train) 
 
# Make predictions 
y_pred = clf.predict(X_test) 
 
# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy * 100:.2f}%') 
 
# Optionally, visualize the decision tree 
from sklearn.tree import plot_tree 
import matplotlib.pyplot as plt # type: ignore 
 
plt.figure(figsize=(20,10)) 
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True) 
plt.show()
