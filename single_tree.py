import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_preprocessing import prepare_data
from joblib import load

# Experiment Runnerrrr
def run_single_tree(
    texts, labels, folds,
    ngram_ranges=[(1,1), (1,2)],
    criterion = ["gini", "entropy"],
    max_depth = [None, 5, 10, 20],
    min_samples_split = [2, 5, 10],
    min_samples_leaf = [1, 2, 5],
    min_dfs=[2,4,6],
    use_tfidf=False,
    cv_folds_list=[5,10],
    random_state=42,
    scoring="f1",
    n_jobs=-1,
    save_prefix="s_tree_results"
):
    all_results = []
    all_misclassified = []

    # Main experiment loop
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

                # Train and tune
                grid = GridSearchCV(DecisionTreeClassifier(random_state=42), {"criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf} ,
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
                    "best_params": grid.best_params_,
                    "cv_f1_mean": grid.best_score_,
                    "test_acc": acc,
                    "test_prec": prec,
                    "test_rec": rec,
                    "test_f1": f1,
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                    "vocab_size": len(vec.get_feature_names_out()),
                    "model_obj": best,   # store the trained model
                    "vectorizer": vec    # store its vectorizer
                })

    # Save results

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"./results/{save_prefix}_overview.csv", index=False)
    print(f"\nSaved all experiment results → ./results/{save_prefix}_overview.csv")

    # Dictionary to store best models per ngram type
    best_models_ngram = {}

    for ngram_type in ngram_ranges:
        ngram_str = str(ngram_type)
        df_subset = df_results[df_results["ngram_range"] == ngram_str]
        best_row = df_subset.sort_values(by="test_f1", ascending=False).iloc[0]
        print(f"\nBest model for ngram={ngram_str}:\n{best_row[['min_df', 'cv_folds', 'best_params', 'test_f1']]}")

        best_model = best_row["model_obj"]
        best_vec = best_row["vectorizer"]

        # Feature contributions for the best Decision Tree model
        feature_names = np.array(best_vec.get_feature_names_out())
        tree_ = best_model.tree_

        feature_fake_count = {f: 0 for f in feature_names}
        feature_real_count = {f: 0 for f in feature_names}

        for i in range(tree_.node_count):
            feature_idx = tree_.feature[i]
            if feature_idx != -2:  # not a leaf node
                majority_class = np.argmax(tree_.value[i])
                feature_name = feature_names[feature_idx]
                if majority_class == 0:
                    feature_fake_count[feature_name] += 1
                else:
                    feature_real_count[feature_name] += 1

        top_fake = sorted(feature_fake_count, key=feature_fake_count.get, reverse=True)[:5]
        top_real = sorted(feature_real_count, key=feature_real_count.get, reverse=True)[:5]

        # Save features per ngram type
        df_features = pd.DataFrame({
            "Top_Deceptive": np.pad(top_fake, (0, max(0, len(top_real)-len(top_fake))), constant_values=""),
            "Top_Truthful": np.pad(top_real, (0, max(0, len(top_fake)-len(top_real))), constant_values="")
        })
        df_features.to_csv(f"./results/{save_prefix}_top_features_best_{ngram_type[0]}_{ngram_type[1]}.csv", index=False)
        print(f"Saved top feature contributions for ngram={ngram_str} → ./results/{save_prefix}_top_features_best_{ngram_type[0]}_{ngram_type[1]}.csv")
        print("\nTop 5 Deceptive terms:", ", ".join(top_fake))
        print("Top 5 Truthful terms:", ", ".join(top_real))

        # Save best model info
        best_models_ngram[ngram_str] = {
            "row": best_row,
            "model": best_model,
            "vectorizer": best_vec,
            "top_fake": top_fake,
            "top_real": top_real
        }

    #return everything
    return df_results, best_models_ngram



path = "./reviews.joblib"
texts, labels, folds = load(path)
df_results, best_models_ngram = run_single_tree(texts, labels, folds)
