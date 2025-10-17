import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_preprocessing import prepare_data
from joblib import load
def run_gradient_boosting(
    texts, labels, folds,
    ngram_ranges=[(1,1), (1,2)],
    n_estimators=[50, 100, 150],
    learning_rate=[0.05, 0.1],
    max_depth=[2, 4, 6],
    min_dfs=[1, 2, 5],
    use_tfidf=False,
    cv_folds_list=[5, 10],
    random_state=42,
    n_iter=10,
    scoring="f1",
    n_jobs=-1,
    save_prefix="gb_results"
):
    all_results = []
    
    # Main loop over n-grams, min_df, CV folds
    for ngram_range in ngram_ranges:
        for min_df in min_dfs:
            for cv_folds in cv_folds_list:
                print(f"\n>>> ngram={ngram_range}, min_df={min_df}, cv_folds={cv_folds}")

                # Prepare data
                X_train, y_train, X_test, y_test, vec, cv, test_texts, _ = prepare_data(
                    texts, labels, folds,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    use_tfidf=use_tfidf,
                    cv_folds=cv_folds,
                    random_state=random_state
                )

                # Set up hyperparameter grid
                param_grid = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth
                }

                rand_search = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=random_state),
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state
                )

                rand_search.fit(X_train, y_train)
                best = rand_search.best_estimator_

                
                # Evaluate on test set
                y_pred = best.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                # Store results
                all_results.append({
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "cv_folds": cv_folds,
                    "best_params": rand_search.best_params_,
                    "cv_f1_mean": rand_search.best_score_,
                    "test_acc": acc,
                    "test_prec": prec,
                    "test_rec": rec,
                    "test_f1": f1,
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                    "vocab_size": len(vec.get_feature_names_out()),
                    "model_obj": best,
                    "vectorizer": vec
                })

    # Convert to DataFrame and save
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"./results/{save_prefix}_overview.csv", index=False)
    print(f"\nSaved all experiment results → ./results/{save_prefix}_overview.csv")

    # Identify best model per ngram
    best_models_ngram = {}
    for ngram_type in ngram_ranges:
        ngram_str = str(ngram_type)
        df_subset = df_results[df_results["ngram_range"] == ngram_str]
        best_row = df_subset.sort_values(by="test_f1", ascending=False).iloc[0]
        print(f"\nBest model for ngram={ngram_str}:\n{best_row[['min_df', 'cv_folds', 'best_params', 'test_f1']]}")

        best_models_ngram[ngram_str] = {
            "row": best_row,
            "model": best_row["model_obj"],
            "vectorizer": best_row["vectorizer"]
        }

    return df_results, best_models_ngram


# Load data and run
path = "./reviews.joblib"
texts, labels, folds = load(path)
df_results, best_models_ngram = run_gradient_boosting(texts, labels, folds)
