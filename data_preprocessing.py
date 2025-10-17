from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2


"""
This file contains the method for preprocessing the data, which can be called from other folders in the project.
"""

# This function genereates the train/test split with vectorization. Optionaly it applies the Chi-square feature selection and/or the StratifiedKFold CV splitter.
def prepare_data(texts, labels, folds, ngram_range=(1,1), min_df=2, use_tfidf=True, cv_folds=5, random_state=42, feature_selection=False, k_features=3000):
    """
    Args:
        texts, labels, folds: dataset lists
        ngram_range (tuple): (1,1)=unigrams, (1,2)=unigrams+bigrams
        min_df: remove sparse terms (e.g., 2 keeps only words appearing in >=2 docs)
        use_tfidf (bool): True=TF-IDF, False=raw term frequencies
        cv_folds (int): number of folds for cross-validation
        random_state: seed
        feature_selection (bool): if True, apply Chi-square feature selection (for Naive Bayes)
        k_features (int): number of top features to keep

    Returns:
        X_train, y_train, X_test, y_test, vectorizer, cv_splitter, test_texts, selector (or None)
    """
    # Split folds (1–4 = train, 5 = test)
    train_texts = [t for t, f in zip(texts, folds) if f != "fold5"]
    test_texts  = [t for t, f in zip(texts, folds) if f == "fold5"]
    y_train = [y for y, f in zip(labels, folds) if f != "fold5"]
    y_test  = [y for y, f in zip(labels, folds) if f == "fold5"]

    # Choose vectorizer
    VectorizerClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vectorizer = VectorizerClass(ngram_range=ngram_range, min_df=min_df)

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    # Optional feature selection (only if True)
    selector = None
    if feature_selection:
        k_features = min(k_features, X_train.shape[1])  # safety cap
        selector = SelectKBest(chi2, k=k_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_test  = selector.transform(X_test)
        print(f"Applied Chi-square feature selection: kept {k_features} features")

    # Stratified cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    return X_train, y_train, X_test, y_test, vectorizer, cv_splitter, test_texts, selector
