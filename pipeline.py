import time

import mlflow
import numpy as np
import xgboost as xgb
from feature_engine.encoding import WoEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from prepare_dataset import prepare_data

# merge_rare_org=False: pipeline.py fits WoE on the full 80% X_train, which
# has not been observed to hit the ORGANIZATION_TYPE zero-denominator crash
# at that size 
X_train, X_test, y_train, y_test = prepare_data(merge_rare_fields_across_cat=False)

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
log_cols = [
    'AMT_INCOME_TOTAL',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE',
]
num_cols = [col for col in num_cols if col not in log_cols]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])
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

# Computed once, used both in the model and in the MLflow log - these 
# never diverge.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

with mlflow.start_run():
    start = time.time()
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"Preprocessing: {time.time() - start:.2f}s")

    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train_transformed, y_train)
    print(f"XGBoost fit only: {time.time() - start:.2f}s")

    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("encoding", "WoE")
    mlflow.log_param("log_transform", ",".join(log_cols))
    mlflow.log_param("merge_rare_org", False)
    mlflow.sklearn.log_model(model, name="model")
    mlflow.log_metric("test_auc", test_auc)
    print(f"Test AUC: {test_auc:.4f}")