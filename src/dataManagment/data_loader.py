import csv


class DataLoader:
  def __init__(self, file_path):
    """
    Initialise with the path to the dataset file (CSV).
    """
    self.file_path = file_path

  def load_data(self):
    """
    Loads the CSV dataset and returns the texts and labels.

    Returns:
        texts (list): List of article texts.
        labels (list): Corresponding labels (categories).
    """
    texts = []
    labels = []
    with open(self.file_path, 'r', encoding='utf-8') as file:
      reader = csv.DictReader(file)
      for row in reader:
        texts.append(row['Text'])
        labels.append(row['Category'])
    return texts, labels

  def get_unique_categories(self):
    """
    Loads the unique categories (labels) in the dataset.

    Returns:
        unique_categories (list): Unique category labels.
    """
    labels = set()  # Use a set to store unique labels
    with open(self.file_path, 'r', encoding='utf-8') as file:
      reader = csv.DictReader(file)
      for row in reader:
        labels.add(row['Category'])
    return list(labels)
