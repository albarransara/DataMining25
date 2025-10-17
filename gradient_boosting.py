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
    n_estimators=[150, 200, 300],
    learning_rate=[0.05, 0.1, 0.15],
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

    # Identify best model per ngram and extract important features
    best_models_ngram = {}
    for ngram_type in ngram_ranges:
        ngram_str = str(ngram_type)
        df_subset = df_results[df_results["ngram_range"] == ngram_str]
        best_row = df_subset.sort_values(by="test_f1", ascending=False).iloc[0]
        print(f"\nBest model for ngram={ngram_str}:\n{best_row[['min_df', 'cv_folds', 'best_params', 'test_f1']]}")

        best_model = best_row["model_obj"]
        vectorizer = best_row["vectorizer"]
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Get feature importances
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top_features = feature_names[sorted_idx][:20]
        top_importances = importances[sorted_idx][:20]

        print("\nTop 10 most important features overall:")
        for feat, imp in zip(top_features[:10], top_importances[:10]):
            print(f"{feat:20s} {imp:.4f}")

        # === Estimate feature direction (fake vs genuine) ===
        # Get test data again for this combination
        X_train, y_train, X_test, y_test, vec, cv, test_texts, _ = prepare_data(
            texts, labels, folds,
            ngram_range=ngram_type,
            min_df=best_row["min_df"],
            use_tfidf=use_tfidf,
            cv_folds=int(best_row["cv_folds"]),
            random_state=random_state
        )

        y_prob = best_model.predict_proba(X_test)[:, 1]  # probability of genuine
        X_array = X_test.toarray()

        feature_effects = []
        for i, f in enumerate(feature_names):
            mask = X_array[:, i] > 0
            if mask.sum() > 5:  # only consider words that appear enough times
                avg_with = y_prob[mask].mean()
                avg_without = y_prob[~mask].mean()
                effect = avg_with - avg_without
                feature_effects.append((f, effect, importances[i]))

        # Build DataFrame
        df_feat = pd.DataFrame(feature_effects, columns=["feature", "effect", "importance"])
        df_feat = df_feat.sort_values(by="importance", ascending=False)

        # Determine fake vs genuine indicators
        top_fake = df_feat.sort_values(by=["effect", "importance"], ascending=[True, False]).head(5)
        top_genuine = df_feat.sort_values(by=["effect", "importance"], ascending=[False, False]).head(5)

        print("\nTop 5 terms pointing toward FAKE reviews:")
        print(top_fake[["feature", "effect", "importance"]].to_string(index=False))

        print("\nTop 5 terms pointing toward GENUINE reviews:")
        print(top_genuine[["feature", "effect", "importance"]].to_string(index=False))

        # Save feature importance file per n-gram type
        filename = f"./results/{save_prefix}_feature_importance_{ngram_type[0]}_{ngram_type[1]}.csv"
        df_feat.to_csv(filename, index=False)
        print(f"Saved feature importance analysis → {filename}")

        # Save best model info
        best_models_ngram[ngram_str] = {
            "row": best_row,
            "model": best_model,
            "vectorizer": vectorizer,
            "top_fake": top_fake,
            "top_genuine": top_genuine
        }

    return df_results, best_models_ngram


# Load data and run
path = "./reviews.joblib"
texts, labels, folds = load(path)
df_results, best_models_ngram = run_gradient_boosting(texts, labels, folds)
