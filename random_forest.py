from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from joblib import load
from data_preprocessing import prepare_data
import pandas as pd
import numpy as np


def run_logistic_regression(texts, labels, folds, ngram_ranges=[(1,1), (1,2)],  min_dfs=[0., 2, 5, 10, 50],  random_state=42, scoring="f1", n_jobs=-1, use_tfidf=True, save_prefix="rf_results"):

        """
        Args:
            texts, labels, folds: dataset lists
            ngram_range (tuple): (1,1)=unigrams, (1,2)=unigrams+bigrams
            min_df: remove sparse terms (e.g., 2 keeps only words appearing in >=2 docs)
            random_state: by fixing a number we ensure reproducability
            save_prefix: name of the files where results will be saved 
            scoring : reference metric during hyperparameter tunning
            use_tfidf: if True the text will get vectorized with TfidfVectorizer, else with CountVectorizer
            n_jobs : number of jobs to run in parallel during hyperparameter tunning, -1 uses all processors
        Returns:
            
        """
        print("----- Random Forest -----")

        # Define paramters we will consider in teh hyperparameter tunning
        param_grid = { "n_estimators": [100, 300, 500], "criterion": ["gini", "entropy", "log_loss"], "max_depth": [None, 10, 50],"max_features": [None, "sqrt", "log2"]}  
        all_results = []
        all_misclassified = []

        for ngram_range in ngram_ranges: # train and evalute with and without bigrams
            for min_df in min_dfs: # try different miniumum frequencies to remove the sparse terms 
                print(f"\n>>> ngram={ngram_range}, min_df={min_df}")

                # Prepare data
                X_train, y_train, X_test, y_test, vec, cv, test_texts, _ = prepare_data(texts, labels, folds, ngram_range=ngram_range, min_df=min_df, use_tfidf=use_tfidf, random_state=random_state)

                # Manual OOB-based hyperparameter tuning
                best_model, best_params, best_oob = None, None, -np.inf

                for params in ParameterGrid(param_grid):
                    model = RandomForestClassifier(oob_score=True,n_jobs=n_jobs, random_state=random_state, **params)
                    model.fit(X_train, y_train)
                    oob = model.oob_score_

                    print(f"Params={params}, OOB={oob:.4f}")

                    if oob > best_oob:
                        best_oob = oob
                        best_model = model
                        best_params = params

                print(f"\nBest OOB params for ngram={ngram_range}, min_df={min_df}: {best_params}, OOB={best_oob:.4f}")

                # Evaluate best model on test data
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                # Misclassified examples
                mis_idx = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
                for i in mis_idx:
                    all_misclassified.append({
                        "ngram_range": str(ngram_range),
                        "min_df": min_df,
                        "true_label": y_test[i],
                        "pred_label": y_pred[i],
                        "text": test_texts[i]
                    })

                # Save metrics
                all_results.append({
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "best_params": best_params,
                    "oob_score": best_oob,
                    "test_acc": acc,
                    "test_prec": prec,
                    "test_rec": rec,
                    "test_f1": f1,
                    "test_auc": auc,
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                    "vocab_size": len(vec.get_feature_names_out()),
                    "model_obj": best_model,
                    "vectorizer": vec
                })

        # Save results
        df_results = pd.DataFrame(all_results).sort_values(by="test_f1", ascending=False).reset_index(drop=True)
        df_results.to_csv(f"./results/{save_prefix}_overview.csv", index=False)
        print(f"\nSaved all experiment results → ./results/{save_prefix}_overview.csv")

        # Best model
        best_row = df_results.iloc[0]
        print(f"\nBest model configuration:\n{best_row[['ngram_range', 'min_df', 'test_f1']]}")

        best_model = best_row["model_obj"]
        best_vec = best_row["vectorizer"]

        # Save misclassified samples for best model
        df_misclassified = pd.DataFrame(all_misclassified)
        df_best_mis = df_misclassified[
            (df_misclassified["ngram_range"] == best_row["ngram_range"]) &
            (df_misclassified["min_df"] == best_row["min_df"])
        ]
        df_best_mis.to_csv(f"./results/{save_prefix}_misclassified_best.csv", index=False)
        print(f"Saved misclassified reviews for best model → ./results/{save_prefix}_misclassified_best.csv")

        return df_results, df_best_mis

                

path = "./reviews.joblib"
texts, labels, folds = load(path)
df_results, df_best_mis = run_logistic_regression(texts, labels, folds)


