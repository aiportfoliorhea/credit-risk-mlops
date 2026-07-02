#import numpy as np
#  print(df.dtypes.value_counts())
# print(df.select_dtypes(include=['object']).columns.tolist())
# print(df.isnull().sum().sort_values(ascending=False).head(20))


# skew = X.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
# print(skew.head(10))

# print(df.isnull().sum().sort_values(ascending=False).head(20))

# Cardinality of categoricals
# print(df.select_dtypes(include=['object']).nunique().sort_values(ascending=False))
# print(df['CODE_GENDER'].value_counts())

# Flag document columns
# flag_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
# print(df[flag_cols].mean().sort_values())