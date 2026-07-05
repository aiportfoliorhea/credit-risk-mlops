import pandas as pd
from sklearn.model_selection import train_test_split
def prepare_data(csv_path='data/application_train.csv', test_size=0.2,
                  random_state=42, merge_rare_org=False):
    df = pd.read_csv(csv_path)
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df.drop(columns=['OWN_CAR_AGE'])

    flag_doc_drop = [c for c in df.columns if 'FLAG_DOCUMENT' in c
                      and c not in ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']]
    df = df.drop(columns=flag_doc_drop)

    avg_medi_drop = [c for c in df.columns if c.endswith('_AVG') or c.endswith('_MEDI')]
    df = df.drop(columns=avg_medi_drop)

    df = df[df['CODE_GENDER'] != 'XNA'].reset_index(drop=True)

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

    # ORGANIZATION_TYPE rare-category merge (threshold: <100 rows in full X_train).
    # Only needed by train_final_model.py, which fits WoE on a reduced ~68%
    # split (X_train minus the early-stopping val carve-out) — small enough
    # for 'Trade: type 5' (31 rows) and 6 similarly-thin categories to hit a
    # WoE zero-denominator crash. pipeline.py (fits on full 80% X_train) and
    # tune_hyperparams.py (fits per-CV-fold, ~53% per fold) have not been
    # observed to hit this failure at their respective fit sizes, so the
    # merge is opt-in rather than global — do not enable without first
    # confirming the same crash occurs at that fit size.
    if merge_rare_org:
        org_counts = X_train['ORGANIZATION_TYPE'].value_counts()
        rare_orgs = org_counts[org_counts < 100].index.tolist()
        dominant_org = org_counts.idxmax()
        X_train['ORGANIZATION_TYPE'] = X_train['ORGANIZATION_TYPE'].replace(rare_orgs, dominant_org)
        X_test['ORGANIZATION_TYPE'] = X_test['ORGANIZATION_TYPE'].replace(rare_orgs, dominant_org)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test