import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Load Data
train_df = pd.read_csv("hacktrain.csv").drop(columns=["Unnamed: 0"], errors="ignore")
test_df = pd.read_csv("hacktest.csv").drop(columns=["Unnamed: 0"], errors="ignore")

X_train = train_df.drop(columns=["ID", "class"])
y_train = train_df["class"]
X_test = test_df.drop(columns=["ID"])
test_ids = test_df["ID"]

# 2. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)

# 3. Feature Engineering
def advanced_features(df):
    ndvi_cols = [c for c in df.columns if '_N' in c]
    df["ndvi_mean"] = df[ndvi_cols].mean(axis=1)
    df["ndvi_std"] = df[ndvi_cols].std(axis=1)
    df["ndvi_min"] = df[ndvi_cols].min(axis=1)
    df["ndvi_max"] = df[ndvi_cols].max(axis=1)
    df["ndvi_range"] = df["ndvi_max"] - df["ndvi_min"]
    df["ndvi_median"] = df[ndvi_cols].median(axis=1)
    df["ndvi_iqr"] = df[ndvi_cols].quantile(0.75, axis=1) - df[ndvi_cols].quantile(0.25, axis=1)
    df["ndvi_slope"] = df[ndvi_cols].apply(lambda row: pd.Series(row).interpolate().diff().mean(), axis=1)
    df["ndvi_nan_count"] = df[ndvi_cols].isna().sum(axis=1)
    df["ndvi_skew"] = df[ndvi_cols].skew(axis=1)
    df["ndvi_kurt"] = df[ndvi_cols].kurt(axis=1)
    return df

X_train = advanced_features(X_train)
X_test = advanced_features(X_test)

# 4. Build Pipeline
pipeline = make_pipeline(
    KNNImputer(n_neighbors=5),
    StandardScaler(),
    SelectKBest(score_func=f_classif, k=30),  # select top 30 features
    LogisticRegression(max_iter=4000, C=10, multi_class="multinomial", solver="lbfgs")
)

# 5. Fit Model and Predict
pipeline.fit(X_train, y_encoded)
y_pred = pipeline.predict(X_test)
predicted_labels = label_encoder.inverse_transform(y_pred)

# 6. Save Submission
final_df = pd.DataFrame({
    "ID": test_ids,
    "class": predicted_labels
})
final_df.to_csv("final_submission_optimized.csv", index=False)
print("✅ Submission saved as 'final_submission_optimized.csv'")
