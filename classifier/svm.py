import numpy as np


class SVM(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, train_iterations=100, learning_rate=1e-3, regularization=1e-5, batch_size=200, verbose=False):
        records, features = X.shape
        num_classes = np.max(y) + 1
        if not self.W:
            self.W = 5e-5 * np.random.randn(features, num_classes)

        # Run stochastic gradient descent to optimize W
        losses = []
        for itr in range(train_iterations):
            mini_batch_indices = np.random.choice(records, batch_size, replace=True)
            X_batch = X[mini_batch_indices]
            y_batch = y[mini_batch_indices]

            loss, gradient = self.loss(X_batch, y_batch, regularization)
            losses.append(loss)

            self.W = self.W + learning_rate * (-gradient)
            if verbose:
                if itr % 100 == 0:
                    print(f"iteration {itr} / {train_iterations}: loss {loss}")

        return losses

    def predict(self, X):
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred

    def loss(self, batch_features, batch_target, regularization):
        records = batch_features.shape[0]
        classes = self.W.shape[1]
        gradient = np.zeros((records, classes))

        scores = np.dot(batch_features, self.W)
        correct_class_score = scores[range(records), list(batch_target)].reshape(-1, 1)
        margin = np.maximum(0, scores - correct_class_score + 1)
        margin[range(records), list(batch_target)] = 0
        loss = np.sum(margin) / records + regularization * np.sum(self.W * self.W)

        gradient[margin > 0] = 1
        gradient[range(records), list(batch_target)] = 0
        gradient[range(records), list(batch_target)] = -np.sum(gradient, axis=1)
        dW = np.dot(batch_features.T, gradient) / records + regularization * self.W

        return loss, dW
