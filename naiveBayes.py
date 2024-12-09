import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Parameters:
        X: np.ndarray or pd.DataFrame - Features
        y: np.ndarray or pd.Series - Target labels
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Calculate class priors P(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.class_priors[cls] = X_c.shape[0] / n_samples

            # Calculate likelihood P(x|y) for each feature
            self.feature_probs[cls] = {}
            for feature_idx in range(n_features):
                feature_values = np.unique(X[:, feature_idx])
                self.feature_probs[cls][feature_idx] = {}

                for value in feature_values:
                    # P(x|y) = Count of feature value in class / Total samples in class
                    self.feature_probs[cls][feature_idx][value] = (
                            np.sum(X_c[:, feature_idx] == value) / X_c.shape[0]
                    )

    def predict_proba(self, X):
        """
        Predict probabilities for each class.

        Parameters:
        X: np.ndarray or pd.DataFrame - Features

        Returns:
        np.ndarray - Predicted probabilities for each class
        """
        n_samples, n_features = X.shape
        probabilities = np.zeros((n_samples, len(self.classes)))

        for idx, sample in enumerate(X):
            for cls_idx, cls in enumerate(self.classes):
                # Start with the prior probability P(y)
                prob = np.log(self.class_priors[cls])

                # Add log likelihoods P(x|y) for each feature
                for feature_idx in range(n_features):
                    feature_value = sample[feature_idx]
                    feature_probs = self.feature_probs[cls][feature_idx]

                    # Use Laplace smoothing to handle unseen values
                    likelihood = feature_probs.get(feature_value, 1e-6)
                    prob += np.log(likelihood)

                probabilities[idx, cls_idx] = prob

        # Exponentiate to get probabilities (and normalize for stability)
        probabilities = np.exp(probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X):
        """
        Predict the class labels.

        Parameters:
        X: np.ndarray or pd.DataFrame - Features

        Returns:
        np.ndarray - Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


