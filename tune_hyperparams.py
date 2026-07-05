import time

from flask import json
import numpy as np
from prepare_dataset import prepare_data
import xgboost as xgb
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from feature_engine.encoding import WoEEncoder

# ---- Data loading + cleaning (same as pipeline.py) ----
X_train, X_test, y_train, y_test = prepare_data(merge_rare_fields_across_cat=True)

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
log_cols = [
    'AMT_INCOME_TOTAL', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
]
num_cols = [col for col in num_cols if col not in log_cols]

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('woe', WoEEncoder()),
])
log_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, validate=False)),
])

# ---- Optuna objective ----
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    # Split on raw X_train/y_train (DataFrame/Series), NOT the pre-transformed
    # array. Fold indices must select from unencoded data so the preprocessor
    # can be fit fresh on each fold's train portion only.
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr_raw = X_train.iloc[train_idx]
        X_val_raw = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        # Rebuild + refit preprocessor on THIS fold's train data only.
        # cat_cols/num_cols/log_cols are fixed column-name lists computed
        # once outside the loop (schema doesn't change per fold), but the
        # transformer objects themselves must be fresh per fold — WoE
        # statistics are supervised and must never see val_idx rows.
        fold_preprocessor = ColumnTransformer([
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
            ('log', log_transformer, log_cols),
        ], remainder='passthrough')

        X_tr = fold_preprocessor.fit_transform(X_tr_raw, y_tr)
        X_val = fold_preprocessor.transform(X_val_raw)

        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            early_stopping_rounds=20,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))

    return np.mean(auc_scores)

if __name__ == "__main__":
    start = time.time()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    print(f"Search time: {time.time() - start:.1f}s")
    print("Best AUC:", study.best_value)
    print("Best params:", study.best_params)
    # Persist results so they survive beyond this terminal session.
    results = {
        "best_auc": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "search_time_s": round(elapsed, 1),
    }
    with open("optuna_best_params_increased_n_est.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to optuna_best_params.json")