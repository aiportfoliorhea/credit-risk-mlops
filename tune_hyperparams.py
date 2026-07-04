import time
import pandas as pd
import numpy as np
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
df = pd.read_csv('data/application_train.csv')
df = df[df['CODE_GENDER'] != 'XNA']
df = df.drop(columns=['OWN_CAR_AGE'])
flag_doc_drop = [c for c in df.columns if 'FLAG_DOCUMENT' in c and c not in ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']]
df = df.drop(columns=flag_doc_drop)
avg_medi_drop = [c for c in df.columns if c.endswith('_AVG') or c.endswith('_MEDI')]
df = df.drop(columns=avg_medi_drop)
df = df[df['CODE_GENDER'] != 'XNA'].reset_index(drop=True)

X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rare_income = ['Unemployed', 'Student', 'Businessman', 'Maternity leave']
X_train['NAME_INCOME_TYPE'] = X_train['NAME_INCOME_TYPE'].replace(rare_income, 'Working')
X_test['NAME_INCOME_TYPE'] = X_test['NAME_INCOME_TYPE'].replace(rare_income, 'Working')
X_train['NAME_FAMILY_STATUS'] = X_train['NAME_FAMILY_STATUS'].replace('Unknown', 'Married')
X_test['NAME_FAMILY_STATUS'] = X_test['NAME_FAMILY_STATUS'].replace('Unknown', 'Married')

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

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

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
    ('log', log_transformer, log_cols),
], remainder='passthrough')

X_train_transformed = preprocessor.fit_transform(X_train, y_train)
X_test_transformed = preprocessor.transform(X_test)

# ---- Optuna objective ----
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_train_transformed, y_train):
        X_tr, X_val = X_train_transformed[train_idx], X_train_transformed[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

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
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print(f"Search time: {time.time() - start:.1f}s")
    print("Best AUC:", study.best_value)
    print("Best params:", study.best_params)