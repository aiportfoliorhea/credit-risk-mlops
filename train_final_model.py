from sklearn.model_selection import train_test_split as tts
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.encoding import WoEEncoder
import numpy as np
from sklearn.preprocessing import FunctionTransformer

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

best_params = {
    'max_depth': 3,
    'learning_rate': 0.08,
    'n_estimators': 385,
    'min_child_weight': 7,
    'subsample': 0.9302,
    'colsample_bytree': 0.8143,
    'gamma': 2.3196,
}

# Carve a validation split out of X_train for early stopping only.
# X_test remains fully untouched for final evaluation.
X_tr, X_val, y_tr, y_val = tts(
    X_train_transformed, y_train, test_size=0.15, stratify=y_train, random_state=42
)

with mlflow.start_run():
    model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        eval_metric='auc',
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    for k, v in best_params.items():
        mlflow.log_param(k, v)
    mlflow.log_param("best_iteration", model.best_iteration)
    mlflow.log_param("scale_pos_weight", model.scale_pos_weight)
    mlflow.log_param("encoding", "WoE")
    mlflow.log_param("log_transform", ", ".join(log_cols))
    mlflow.sklearn.log_model(model, name="model")
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("val_auc_best_iter", model.best_score)
    print(f"Best iteration: {model.best_iteration} / 385 ceiling")
    print(f"Test AUC: {test_auc:.4f}")