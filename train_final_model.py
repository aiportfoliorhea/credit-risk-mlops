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
from prepare_dataset import prepare_data

# ---- Data loading + cleaning (same as pipeline.py) ----
X_train, X_test, y_train, y_test = prepare_data(merge_rare_fields_across_cat=True)

# ---- Carve val split out of RAW X_train, BEFORE the preprocessor is fit ----
# This must happen before preprocessor.fit_transforsm, otherwise WoE (a
# supervised encoder) leaks y_train's target information into X_val's
# encoding, since X_val is a subset of the data WoE was fit on.
X_tr_raw, X_val_raw, y_tr, y_val = tts(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)
X_tr_raw = X_tr_raw.reset_index(drop=True)
X_val_raw = X_val_raw.reset_index(drop=True)
y_tr = y_tr.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

cat_cols = X_tr_raw.select_dtypes(include=['object']).columns.tolist()
num_cols = X_tr_raw.select_dtypes(include=['number']).columns.tolist()
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


# Fit ONLY on X_tr_raw + y_tr — X_val and X_test are transform-only.
X_tr_transformed = preprocessor.fit_transform(X_tr_raw, y_tr)
X_val_transformed = preprocessor.transform(X_val_raw)
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
feature_engg_best_params = {
    'max_depth': 5, 
    'learning_rate': 0.03616475500528253, 
    'n_estimators': 376, 
    'min_child_weight': 7, 
    'subsample': 0.8811141285123267, 
    'colsample_bytree': 0.6041836057038531, 
    'gamma': 0.6781890863887554
}

# updated n-estimators to 800 with feature engineering updated numbers
# Best params: {'max_depth': 3, 'learning_rate': 0.09964034718600043, 'n_estimators': 377, 'min_child_weight': 8, 'subsample': 0.9742636936312206, 'colsample_bytree': 0.6050501389679301, 'gamma': 2.034350998305848}

with mlflow.start_run(run_name="feature_engg_best_param"):
    model = xgb.XGBClassifier(
        **feature_engg_best_params,
        random_state=42,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        eval_metric='auc',
        early_stopping_rounds=20,
    )
    model.fit(X_tr_transformed, y_tr, eval_set=[(X_val_transformed, y_val)], verbose=False)

    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    for k, v in feature_engg_best_params.items():
        mlflow.log_param(k, v)
    mlflow.log_param("best_iteration", model.best_iteration)
    mlflow.log_param("scale_pos_weight", model.scale_pos_weight)
    mlflow.log_param("encoding", "WoE")
    mlflow.log_param("log_transform", ", ".join(log_cols))
    mlflow.sklearn.log_model(model, name="model")
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("val_auc_best_iter", model.best_score)
    print(f"Best iteration: {model.best_iteration} / {feature_engg_best_params['n_estimators']} ceiling")
    print(f"Test AUC: {test_auc:.4f}")