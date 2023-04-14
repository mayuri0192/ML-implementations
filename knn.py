
import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class KNN:
    def __init__(self,k=3):
        self.k = k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predict_labels = [self._predict(x) for x in X]
        return np.array(predict_labels)
    
    def _predict(self,x):
        distances  = [euclidean_distance(x,x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: 10]
        k_neighbor_label = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_label).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    cmap = ListedColormap(['red','green','blue'])

    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 52)
    k=3
    clf = KNN(k=k)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))