'''
Based  on bias theory which says if we have wo events A and B  
then the probability of event A given that B has already happended 
is equal to the probability of B given that A has happend X probability of A / probability of B

P(A/B) = p(B/A)*p(A)/P(B)

In our case: 

P(y|X) = P(X|y).P(y)/p(X)

our feature vector:
X = (x1,x2,x3 .... xn)
Assuming all features are mutually independent

P(y|x) = P(x1|y)*P(x2|y)......P(Xn|y)*P(y)/P(X)

In real life not all features are mutualy independent

Example: if a person is going for run given that sun is shining and person is healthy 
-- both these features are independednt of each other and contribute to the person running

p(y|xi) - posterier probabilty
P(xi|y) - conditional probability
P(y) - prior probability of y
P(X) - prior probabilty of X

Select class with hightest probability : 
for that we use log fucntion and argmax function
We model the probability wih guassian function
'''


import numpy as np

class NaiveBayes:

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # init mean and vars and priors

        self._mean = np.zeros((n_classes, n_features),dtype = np.float64)
        self._var = np.zeros((n_classes, n_features),dtype = np.float64)
        self._priors = np.zeros(n_classes,dtype = np.float64)

        for c in self._classes:
            X_c  = X[c==y]
            self._mean[c,:] = X_c.mean(axis = 0)
            self._var[c,:] = X_c.var(axis = 0)
            self._priors[c] = X_c.shape[0]/ float(n_samples)


    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    # Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))