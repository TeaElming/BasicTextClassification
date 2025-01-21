# Wikipedia 300 Text Classification

This project classifies 300 Wikipedia articles (150 about Video Games and 150 about Programming) using machine learning.

## Dataset
- The dataset is split into two folders: `data/programming` and `data/video_games`.
- Each folder contains text files (articles).

## Requirements
- Python 3.x
- Scikit-learn
- See `requirements.txt` for details.

Install dependencies:
``` pip install -r requirements.txt```


## Running the Project

1. Place your dataset inside the `data` folder under `programming` and `video_games`.
2. Run `python main.py` from within the `src` folder. Adjust the `data_path` if needed.

## Project Structure
- `data/`
   - `programming/` (contains 150 text files)
   - `video_games/` (contains 150 text files)
- `src/`
   - `dataManagement/data_loader.py` (Loads text data and labels)
   - `dataManagement/data_parser.py` (Handles preprocessing and feature extraction (Bag-of-Words or TF-IDF).)
   - `models/classifier.py` (Implements training & evaluation with Scikit-learn)
   - `main.py` (Orchestrates everything)
- `README.md`
- `requirements.txt`

## Levels

- **E level**:
  - Uses Bag-of-Words (CountVectorizer).
  - Trains models and reports accuracy on the same training data.

- **C-D level**:
  - Uses Bag-of-Words and 10-fold cross-validation.

- **A-B level**:
  - Uses TF-IDF (TfidfVectorizer) and 10-fold cross-validation.
  - Checks if TF-IDF improves classification accuracy compared to Bag-of-Words.

## Notes
- The `TextClassifier` class handles both Bag-of-Words and TF-IDF as well as
  either same-set evaluation or cross-validation.
- Modify the code as needed if you want to store or visualise results.
