import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow


df = pd.read_csv('data/application_train.csv')
# print(df.columns.tolist())
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df['TARGET']
X = X.select_dtypes(include=['number'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run():
    model = xgb.XGBClassifier(n_estimators=100, 
                              max_depth=5, 
                              random_state=42,  
                              early_stopping_rounds=10)
                              
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                              verbose=True)
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    # 3. Calculate test AUC
    y_pred_probability = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_probability)
    
    # 4. Log the test metric
    mlflow.log_metric("test_auc", test_auc)
    
    # 5. Log the model artifact manually
    mlflow.xgboost.log_model(model, artifact_path="model")
    
    print(f"Logged run with Test AUC: {test_auc:.4f}")