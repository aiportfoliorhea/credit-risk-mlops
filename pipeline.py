import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.encoding import WoEEncoder
import numpy as np
from sklearn.preprocessing import FunctionTransformer



df = pd.read_csv('data/application_train.csv')

# Pre-pipeline cleaning
df = df[df['CODE_GENDER'] != 'XNA']
df = df.drop(columns=['OWN_CAR_AGE'])
flag_doc_drop = [c for c in df.columns if 'FLAG_DOCUMENT' in c and c not in ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']]
df = df.drop(columns=flag_doc_drop)
avg_medi_drop = [c for c in df.columns if c.endswith('_AVG') or c.endswith('_MEDI')]
df = df.drop(columns=avg_medi_drop)
df = df[df['CODE_GENDER'] != 'XNA'].reset_index(drop=True)
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df['TARGET']
# print(X.shape)

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
    'AMT_INCOME_TOTAL',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE',
]

# print(f"Categorical: {len(cat_cols)}")
# print(f"Numeric: {len(num_cols)}")
# print(f"Log transform: {log_cols}")

# for cat in cat_cols:
#     cross = pd.crosstab(X_train[cat], y_train)
#     zero_cols = (cross == 0).any(axis=1)
#     if zero_cols.any():
#         print(f"\n{cat}:")
#         print(cross[zero_cols])

# print(X_train['NAME_INCOME_TYPE'].value_counts(normalize=True))
# print(X_train['NAME_FAMILY_STATUS'].value_counts(normalize=True))

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

num_cols = [col for col in num_cols if col not in log_cols]

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
    ('log', log_transformer, log_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    ))
])


with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("scale_pos_weight", "auto")
    mlflow.log_param("encoding", "WoE")
    mlflow.log_param("log_transform", "AMT_INCOME_TOTAL, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, OBS_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE")
    mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")
    mlflow.log_metric("test_auc", test_auc)
    
    print(f"Test AUC: {test_auc:.4f}")