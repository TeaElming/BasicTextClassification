from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Classifier:
  def __init__(self, algorithm='naive_bayes'):
    """
    Initialise the classifier.
    Args:
      algorithm (str): The algorithm to use ('naive_bayes' or 'linear_svc').
    """
    if algorithm == 'naive_bayes':
      self.model = MultinomialNB()
    elif algorithm == 'linear_svc':
      self.model = LinearSVC(max_iter=10000)
    else:
      raise ValueError(
          "Algorithm must be either 'naive_bayes' or 'linear_svc' for the classifier to work.")

  def fit(self, X_train, y_train):
    """
    Train the model.
    Args:
      X_train: Training feature matrix.
      y_train: Training labels.
    """
    self.model.fit(X_train, y_train)  # Uses whichever model has been declared

  def predict(self, X):
    """
    Predict labels for the given feature matrix.
    Args:
      X: Feature matrix.
    Returns:
      Predicted labels.
    """
    return self.model.predict(X)

  def evaluate(self, X, y):
    """
    Evaluate the model on a dataset.
    Args:
      X: Feature matrix.
      y: True labels.
    Returns:
      Accuracy score and detailed classification report.
    """
    y_pred = self.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y, y_pred)
    return accuracy, report, confusion_mat

  def cross_validate(self, X, y, folds=10):
    """
    Perform cross-validation.
    Args:
      X: Feature matrix.
      y: Labels.
      folds (int): Number of folds for cross-validation.
    Returns:
      Dictionary with fold scores, mean accuracy, and standard deviation.
    """
    scores = cross_val_score(self.model, X, y, cv=folds)

    # Print accuracy for each fold
    for fold_idx, score in enumerate(scores, start=1):
      print(f"Fold {fold_idx}: Accuracy = {score*100:.2f}%")

    result = {
        "fold_scores": scores,
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
    }
    return result
