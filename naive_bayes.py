import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from data_preprocessing import prepare_data
from joblib import load

def run_naive_bayes(
    texts, labels, folds,
    ngram_ranges=[(1,1), (1,2)],
    min_dfs=[0.0, 2, 4],
    alpha_values=[0.01, 0.1, 0.3, 0.5, 1],
    cv_folds_list=[4, 6, 10],
    use_tfidf=True,
    feature_selection=True,
    k_features=3000,
    random_state=42,
    scoring="f1",
    n_jobs=-1,
    save_prefix="nb_results"
):
    """Naive Bayes experiment with best-model misclassified samples and feature contributions."""
    all_results = []
    all_misclassified = []

    # Main experiment loop
    for ngram_range in ngram_ranges:
        for min_df in min_dfs:
            for cv_folds in cv_folds_list:
                print(f"\n>>> ngram={ngram_range}, min_df={min_df}, cv_folds={cv_folds}, feature_selection={feature_selection}")

                # Prepare data
                X_train, y_train, X_test, y_test, vec, cv, test_texts, _ = prepare_data(
                    texts, labels, folds,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    use_tfidf=use_tfidf,
                    cv_folds=cv_folds,
                    random_state=random_state,
                    feature_selection=feature_selection,
                    k_features=k_features
                )

                # Train and tune
                grid = GridSearchCV(MultinomialNB(), {"alpha": alpha_values},
                                    cv=cv, scoring=scoring, n_jobs=n_jobs)
                grid.fit(X_train, y_train)
                best = grid.best_estimator_

                # Evaluate
                y_pred = best.predict(X_test)
                y_proba = best.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                # Collect misclassified sentences for this combo
                mis_idx = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
                for i in mis_idx:
                    all_misclassified.append({
                        "ngram_range": str(ngram_range),
                        "min_df": min_df,
                        "cv_folds": cv_folds,
                        "true_label": y_test[i],
                        "pred_label": y_pred[i],
                        "text": test_texts[i]
                    })

                # Collect metrics
                all_results.append({
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "cv_folds": cv_folds,
                    "feature_selection": feature_selection,
                    "k_features": k_features if feature_selection else None,
                    "best_alpha": grid.best_params_["alpha"],
                    "cv_f1_mean": grid.best_score_,
                    "test_acc": acc,
                    "test_prec": prec,
                    "test_rec": rec,
                    "test_f1": f1,
                    "test_auc": auc,
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                    "vocab_size": len(vec.get_feature_names_out()),
                    "model_obj": best,   # store the trained model
                    "vectorizer": vec    # store its vectorizer
                })

    # Save results
    df_results = pd.DataFrame(all_results).sort_values(by="test_f1", ascending=False).reset_index(drop=True)
    df_results.to_csv(f"./results/{save_prefix}_overview.csv", index=False)
    print(f"\nSaved all experiment results → ./results/{save_prefix}_overview.csv")

    # Identify the best model (highest F1)
    best_row = df_results.iloc[0]
    print(f"\nBest model configuration:\n{best_row[['ngram_range', 'min_df', 'cv_folds', 'best_alpha', 'test_f1']]}")

    best_model = best_row["model_obj"]
    best_vec = best_row["vectorizer"]

    # Save misclassified samples from the best model only ---
    df_misclassified = pd.DataFrame(all_misclassified)
    df_best_mis = df_misclassified[
        (df_misclassified["ngram_range"] == best_row["ngram_range"]) &
        (df_misclassified["min_df"] == best_row["min_df"]) &
        (df_misclassified["cv_folds"] == best_row["cv_folds"])
    ]
    df_best_mis.to_csv(f"./results/{save_prefix}_misclassified_best.csv", index=False)
    print(f"Saved misclassified reviews for best model → ./results/{save_prefix}_misclassified_best.csv")

    # Feature contributions for the best model
    feature_names = np.array(best_vec.get_feature_names_out())
    log_probs = best_model.feature_log_prob_
    diff = log_probs[1] - log_probs[0]

    top_fake = feature_names[np.argsort(diff)[-10:][::-1]]
    top_real = feature_names[np.argsort(diff)[:10]]

    df_features = pd.DataFrame({
        "Top_Deceptive": np.pad(top_fake, (0, max(0, len(top_real)-len(top_fake))), constant_values=""),
        "Top_Truthful": np.pad(top_real, (0, max(0, len(top_fake)-len(top_real))), constant_values="")
    })
    df_features.to_csv(f"./results/{save_prefix}_top_features_best.csv", index=False)
    print(f"Saved top feature contributions for best model → ./results/{save_prefix}_top_features_best.csv")

    print("\nTop 5 Deceptive terms:", ", ".join(top_fake[:5]))
    print("Top 5 Truthful terms:", ", ".join(top_real[:5]))

    return df_results, df_features, df_best_mis


path = "./reviews.joblib"
texts, labels, folds = load(path)
df_results, df_features, df_best_mis = run_naive_bayes(texts, labels, folds)
