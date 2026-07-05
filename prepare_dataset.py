import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def prepare_data(csv_path='data/application_train.csv', test_size=0.2,
                  random_state=42, merge_rare_fields_across_cat=False):
    df = pd.read_csv(csv_path)
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df.drop(columns=['OWN_CAR_AGE'])

    flag_doc_drop = [c for c in df.columns if 'FLAG_DOCUMENT' in c
                      and c not in ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']]
    df = df.drop(columns=flag_doc_drop)

    avg_medi_drop = [c for c in df.columns if c.endswith('_AVG') or c.endswith('_MEDI')]
    df = df.drop(columns=avg_medi_drop)

    # --- Feature engineering (row-wise only, safe pre-split) ---
    df = add_engineered_features(df)

    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rare_income = ['Unemployed', 'Student', 'Businessman', 'Maternity leave']
    X_train['NAME_INCOME_TYPE'] = X_train['NAME_INCOME_TYPE'].replace(rare_income, 'Working')
    X_test['NAME_INCOME_TYPE'] = X_test['NAME_INCOME_TYPE'].replace(rare_income, 'Working')

    X_train['NAME_FAMILY_STATUS'] = X_train['NAME_FAMILY_STATUS'].replace('Unknown', 'Married')
    X_test['NAME_FAMILY_STATUS'] = X_test['NAME_FAMILY_STATUS'].replace('Unknown', 'Married')

    # ORGANIZATION_TYPE and NAME_EDUCATION_TYPE rare-category merge (threshold: <100 rows in full X_train).
    # Only needed by train_final_model.py, which fits WoE on a reduced ~68%
    # split (X_train minus the early-stopping val carve-out) and
    # tune_hyperparams.py (fits per-CV-fold, ~53% per fold) - small enough
    # for 'Trade: type 5' (31 rows) and 6 similarly-thin categories to hit a
    # WoE zero-denominator crash. pipeline.py (fits on full 80% X_train) so the error
    # isn't observed there.
    if merge_rare_fields_across_cat:
        org_counts = X_train['ORGANIZATION_TYPE'].value_counts()
        rare_orgs = org_counts[org_counts < 100].index.tolist()
        dominant_org = org_counts.idxmax()
        X_train['ORGANIZATION_TYPE'] = X_train['ORGANIZATION_TYPE'].replace(rare_orgs, dominant_org)
        X_test['ORGANIZATION_TYPE'] = X_test['ORGANIZATION_TYPE'].replace(rare_orgs, dominant_org)
        X_train['NAME_EDUCATION_TYPE'] = X_train['NAME_EDUCATION_TYPE'].replace('Academic degree', 'Higher education')
        X_test['NAME_EDUCATION_TYPE'] = X_test['NAME_EDUCATION_TYPE'].replace('Academic degree', 'Higher education')


    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test



def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 11 engineered features. All transforms are row-wise (no cross-row
    statistics), so calling this pre-split is safe with no leakage.
 
    Requires: AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE,
    DAYS_EMPLOYED, DAYS_BIRTH, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3.

    """
    required = [
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED", "DAYS_BIRTH",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_engineered_features missing required columns: {missing}")
 
    df = df.copy()
 
    # 1. DAYS_EMPLOYED anomaly flag + cleanup.
    #    365243 is Home Credit's sentinel for "not employed" (pensioners etc.).
    #    Flag before nulling so the anomaly itself is a usable signal.
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
 
    # 2. Credit-to-income ratio - total leverage.
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
 
    # 3. Annuity-to-income ratio - monthly debt service burden.
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
 
    # 4. Credit-to-annuity ratio - implied loan duration in payment periods.
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
 
    # 5. Employment length as fraction of age - stability signal.
    #    NaN where DAYS_EMPLOYED was nulled (pensioners) — intentional.
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
 
    # 6. Credit vs goods price — financing markup / cash-out signal.
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
 
    # 7. Income per family member.
    if "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_PERSON"] = (
            df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
        )
    else:
        df["INCOME_PER_PERSON"] = np.nan
 
    # 8-10. EXT_SOURCE aggregates.
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
 
    # 11. Count of missing EXT_SOURCE values — missingness itself is informative.
    df["EXT_SOURCE_MISSING_COUNT"] = df[ext_cols].isna().sum(axis=1)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
 