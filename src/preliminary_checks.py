import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb


def nested_cv_holdout_xgboost(df, label_col="mutation_status", n_splits=5):
    label_mapping = {"WT": 0, "MUT": 1}
    df = df.copy()
    df[label_col] = df[label_col].map(label_mapping)

    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    outer_splits = list(outer_skf.split(df.drop(columns=[label_col]), df[label_col]))

    results = []

    for i in range(n_splits):
        print(f"Outer fold {i+1}/{n_splits} (test block)")

        test_idx = outer_splits[i][1]
        test_df = df.iloc[test_idx]
        trainval_df = df.drop(df.index[test_idx])

        X_test = test_df.drop(columns=[label_col])
        y_test = test_df[label_col].values

        inner_skf = StratifiedKFold(n_splits=n_splits-1, shuffle=True, random_state=42)
        inner_X = trainval_df.drop(columns=[label_col])
        inner_y = trainval_df[label_col].values

        inner_acc = []
        inner_f1 = []

        for train_idx, val_idx in inner_skf.split(inner_X, inner_y):
            X_train = inner_X.iloc[train_idx]
            y_train = inner_y[train_idx]
            X_val = inner_X.iloc[val_idx]
            y_val = inner_y[val_idx]

            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            inner_acc.append(accuracy_score(y_val, preds))
            inner_f1.append(f1_score(y_val, preds))

        final_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        final_model.fit(trainval_df.drop(columns=[label_col]), trainval_df[label_col])
        final_preds = final_model.predict(X_test)
        final_acc = accuracy_score(y_test, final_preds)
        final_f1 = f1_score(y_test, final_preds)

        results.append({
            "Outer Fold": i+1,
            "Mean Inner Accuracy": np.mean(inner_acc),
            "Mean Inner F1": np.mean(inner_f1),
            "Test Accuracy": final_acc,
            "Test F1": final_f1
        })

    return pd.DataFrame(results)



def xgboost_balanced_subsample(df, label_col="mutation_status", n_per_class=1000, n_repeats=3, test_size=0.2):
    label_mapping = {"WT": 0, "MUT": 1}
    df = df.copy()
    df[label_col] = df[label_col].map(label_mapping)

    all_results = []

    for run in range(n_repeats):
        print(f"Run {run + 1}/{n_repeats}")

        df_mut = df[df[label_col] == 1].sample(n=n_per_class, random_state=run)
        df_wt = df[df[label_col] == 0].sample(n=n_per_class, random_state=run)
        df_balanced = pd.concat([df_mut, df_wt]).sample(frac=1, random_state=run).reset_index(drop=True)

        X = df_balanced.drop(columns=[label_col])
        y = df_balanced[label_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=run)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=run)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        all_results.append({
            "Run": run + 1,
            "Accuracy": acc,
            "F1 Score": f1
        })

    return pd.DataFrame(all_results)