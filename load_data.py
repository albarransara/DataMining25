import os
import glob
from joblib import dump


"""
This file contains the method to load the orginal dataset. By running the document you can generate a compressed file with the extracted review's
text, labels and corresponding folds. The compressed document gets saved in the project root directory, make in it accessible for the other python files.

NOTE: If you already have the compressed "joblib" file in your directory, you can skip this document.
"""

#  The following function loads review texts, labels, and fold names from a labeled dataset directory.
def load_reviews_simple(base_path):

    texts, labels, folds = [], [], []

    for label_name, label_val in [("deceptive_from_MTurk", 1),
                                  ("truthful_from_Web", 0)]:
        path = os.path.join(base_path, label_name)

        for fold_name in sorted(os.listdir(path)):
            fold_path = os.path.join(path, fold_name)
            if not os.path.isdir(fold_path):
                continue

            for file in glob.glob(os.path.join(fold_path, "*.txt")):
                with open(file, "r", encoding="utf-8") as f:
                    texts.append(f.read().strip())
                labels.append(label_val)
                folds.append(fold_name)

    return texts, labels, folds

base_path = "./Data/negative_polarity" 
destination_path = "./reviews.joblib"
texts, labels, folds = load_reviews_simple(base_path) # Obtain processed arrays with text, labels and folds
dump((texts, labels, folds), destination_path) # Compress them in a joblib file

