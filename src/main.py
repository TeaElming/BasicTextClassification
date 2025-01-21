import time
from dataManagment.data_loader import DataLoader
from dataManagment.data_parser import DataParser
from models.classifier import Classifier
from utils.printing import Printing


def main():
  # Step 1: Load data
  file_path = "data/wikipedia_300.csv"
  loader = DataLoader(file_path)
  texts, labels = loader.load_data()
  print(f"Loaded {len(texts)} articles with {
        len(set(labels))} unique categories.")

  # Step 2: Preprocess data
  bow_parser = DataParser(use_tfidf=False)
  tfidf_parser = DataParser(use_tfidf=True)
  bow_features = bow_parser.fit_transform(texts)
  tfidf_features = tfidf_parser.fit_transform(texts)

  print("Data preprocessing completed.")
  print(f"BoW matrix dimensions: {bow_features.shape}")
  print(f"TF-IDF matrix dimensions: {tfidf_features.shape}")

  # Determine sorted list of class labels for consistent printing
  sorted_labels = sorted(set(labels))

  # Initialize classifiers
  nb_classifier = Classifier(algorithm='naive_bayes')
  svc_classifier = Classifier(algorithm='linear_svc')

  # -------------------- E-level Implementation --------------------
  print("\n--- E-level Implementation ---")

  # Naive Bayes with BoW
  start_time = time.time()
  nb_classifier.fit(bow_features.toarray(), labels)
  nb_bow_accuracy, nb_bow_report, nb_bow_cm = nb_classifier.evaluate(
      bow_features.toarray(), labels)
  end_time = time.time()

  print(f"\nNaive Bayes (BoW): {nb_bow_accuracy * 100:.2f}% accuracy")
  # Use Printing class
  Printing.print_confusion_matrix(nb_bow_cm, sorted_labels)
  Printing.print_classification_metrics(nb_bow_report, sorted_labels)
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # LinearSVC with BoW
  start_time = time.time()
  svc_classifier.fit(bow_features, labels)
  svc_bow_accuracy, svc_bow_report, svc_bow_cm = svc_classifier.evaluate(
      bow_features, labels)
  end_time = time.time()

  print(f"LinearSVC (BoW): {svc_bow_accuracy * 100:.2f}% accuracy")
  Printing.print_confusion_matrix(svc_bow_cm, sorted_labels)
  Printing.print_classification_metrics(svc_bow_report, sorted_labels)
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # -------------------- C-level Implementation --------------------
  print("\n--- C-level Implementation ---")

  # Naive Bayes with BoW Cross-Validation
  start_time = time.time()
  nb_bow_cv_scores = nb_classifier.cross_validate(
      bow_features.toarray(), labels)
  end_time = time.time()
  print(f"Naive Bayes Cross-Validation (BoW): {nb_bow_cv_scores['mean_accuracy'] * 100:.2f}% ± {
        nb_bow_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # LinearSVC with BoW Cross-Validation
  start_time = time.time()
  svc_bow_cv_scores = svc_classifier.cross_validate(bow_features, labels)
  end_time = time.time()
  print(f"LinearSVC Cross-Validation (BoW): {svc_bow_cv_scores['mean_accuracy'] * 100:.2f}% ± {
        svc_bow_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # -------------------- A-level Implementation --------------------
  print("\n--- A-level Implementation ---")

  # Naive Bayes with TF-IDF
  start_time = time.time()
  nb_classifier.fit(tfidf_features.toarray(), labels)
  nb_tfidf_accuracy, nb_tfidf_report, nb_tfidf_cm = nb_classifier.evaluate(
      tfidf_features.toarray(), labels)
  end_time = time.time()

  print(f"Naive Bayes (TF-IDF): {nb_tfidf_accuracy * 100:.2f}% accuracy")
  Printing.print_confusion_matrix(nb_tfidf_cm, sorted_labels)
  Printing.print_classification_metrics(nb_tfidf_report, sorted_labels)
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # LinearSVC with TF-IDF
  start_time = time.time()
  svc_classifier.fit(tfidf_features, labels)
  svc_tfidf_accuracy, svc_tfidf_report, svc_tfidf_cm = svc_classifier.evaluate(
      tfidf_features, labels)
  end_time = time.time()

  print(f"LinearSVC (TF-IDF): {svc_tfidf_accuracy * 100:.2f}% accuracy")
  Printing.print_confusion_matrix(svc_tfidf_cm, sorted_labels)
  Printing.print_classification_metrics(svc_tfidf_report, sorted_labels)
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # Naive Bayes with TF-IDF Cross-Validation
  start_time = time.time()
  nb_tfidf_cv_scores = nb_classifier.cross_validate(
      tfidf_features.toarray(), labels)
  end_time = time.time()
  print(f"Naive Bayes Cross-Validation (TF-IDF): {nb_tfidf_cv_scores['mean_accuracy'] * 100:.2f}% ± {
        nb_tfidf_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # LinearSVC with TF-IDF Cross-Validation
  start_time = time.time()
  svc_tfidf_cv_scores = svc_classifier.cross_validate(tfidf_features, labels)
  end_time = time.time()
  print(f"LinearSVC Cross-Validation (TF-IDF): {svc_tfidf_cv_scores['mean_accuracy'] * 100:.2f}% ± {
        svc_tfidf_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"Execution Time: {end_time - start_time:.2f} seconds.\n")

  # -------------------- Summary --------------------
  print("\n--- Summary ---")
  print(f"Naive Bayes (BoW): {nb_bow_accuracy * 100:.2f}%, CV: {
        nb_bow_cv_scores['mean_accuracy'] * 100:.2f}% ± {nb_bow_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"LinearSVC (BoW): {svc_bow_accuracy * 100:.2f}%, CV: {
        svc_bow_cv_scores['mean_accuracy'] * 100:.2f}% ± {svc_bow_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"Naive Bayes (TF-IDF): {nb_tfidf_accuracy * 100:.2f}%, CV: {
        nb_tfidf_cv_scores['mean_accuracy'] * 100:.2f}% ± {nb_tfidf_cv_scores['std_accuracy'] * 100:.2f}%")
  print(f"LinearSVC (TF-IDF): {svc_tfidf_accuracy * 100:.2f}%, CV: {
        svc_tfidf_cv_scores['mean_accuracy'] * 100:.2f}% ± {svc_tfidf_cv_scores['std_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
  main()
