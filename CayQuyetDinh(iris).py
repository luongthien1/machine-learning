from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=1)

tree = DecisionTreeClassifier()

models = tree.fit(X_train, y_train)

print(models.predict(X_test))
print(y_test)