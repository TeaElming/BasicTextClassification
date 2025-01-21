class Printing:
  @staticmethod
  def _create_separator(widths, corner='+', fill='-'):
    """
    Create a row separator line given column widths.
    E.g. if widths=[10, 8], result = '+----------+--------+'
    """
    line = corner
    for w in widths:
      line += (fill * (w + 2)) + corner
    return line

  @staticmethod
  def _create_row(values, widths, corner='|'):
    """
    Create a single row with vertical bars.
    E.g. if values=['[Games]', '0.95'], widths=[10, 8],
    the result might be '| [Games]   |     0.95 |'
    """
    row = corner
    for val, w in zip(values, widths):
      row += f" {val:<{w}} {corner}"
    return row

  @staticmethod
  def print_confusion_matrix(cm, class_labels):
    """
    Print the confusion matrix in a tabular format with ASCII borders.
    """
    print("Confusion Matrix:")

    max_label_len = max(len(lbl) for lbl in class_labels)
    label_col_width = max_label_len + 2  # row-label column
    value_col_width = 8  # numeric columns

    # Prepare the column widths array
    # First column = row labels, rest are numeric columns for each class
    col_widths = [label_col_width] + [value_col_width] * len(class_labels)

    # Build header row
    header_values = [''] + [f'[{lbl}]' for lbl in class_labels]

    # Print top separator
    print(Printing._create_separator(col_widths))
    # Print header row
    print(Printing._create_row(header_values, col_widths))
    # Print separator after header
    print(Printing._create_separator(col_widths))

    # Print each row of the confusion matrix
    for i, row in enumerate(cm):
      row_values = [f'[{class_labels[i]}]'] + [str(val) for val in row]
      print(Printing._create_row(row_values, col_widths))
      print(Printing._create_separator(col_widths))

    print()  # blank line at the end

  @staticmethod
  def print_classification_metrics(report, class_labels):
    """
    Print precision, recall, and F1-score for each class in a neat ASCII table.
    """
    print("Metrics by category:")

    max_label_len = max(len(lbl) for lbl in class_labels)
    label_col_width = max_label_len + 2
    value_col_width = 9  # for precision, recall, f1

    # Column widths: first for labels, then 3 metrics (precision, recall, f1)
    col_widths = [label_col_width, value_col_width,
                  value_col_width, value_col_width]

    # Header row
    header_values = [
        '',
        'Precision',
        'Recall',
        'F1 score'
    ]

    # Print top separator
    print(Printing._create_separator(col_widths))
    # Print header row
    print(Printing._create_row(header_values, col_widths))
    # Print separator after header
    print(Printing._create_separator(col_widths))

    # Print row for each actual label
    for lbl in class_labels:
      if lbl in report:
        precision = report[lbl]["precision"]
        recall = report[lbl]["recall"]
        f1 = report[lbl]["f1-score"]
        row_values = [
            f'[{lbl}]',
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}"
        ]
        print(Printing._create_row(row_values, col_widths))
        print(Printing._create_separator(col_widths))

    # Weighted avg line
    weighted = report["weighted avg"]
    row_values = [
        'Avg:',
        f"{weighted['precision']:.3f}",
        f"{weighted['recall']:.3f}",
        f"{weighted['f1-score']:.3f}"
    ]
    print(Printing._create_row(row_values, col_widths))
    print(Printing._create_separator(col_widths))
    print()  
