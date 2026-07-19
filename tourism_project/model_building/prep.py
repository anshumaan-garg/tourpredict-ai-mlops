# for data manipulation
import pandas as pd
import os
# for train/test split
from sklearn.model_selection import train_test_split
# for Hugging Face authentication and uploads
from huggingface_hub import HfApi

# Authenticate with the write token from the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the dataset directly from the Hugging Face dataset space
DATASET_PATH = "hf://datasets/NineKnox/tourism-wellness-package/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ------------------------------- Data cleaning -------------------------------
# Drop the unnamed index column (if present) and the unique identifier
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# Fix data-quality issues in categorical columns
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Define target and feature groups
target_col = "ProdTaken"
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# Impute missing values: median for numeric, mode for categorical
for col in numeric_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
for col in categorical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Cleaned dataset shape:", df.shape)

# --------------------------- Train / test split ------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Saved train/test splits locally.")

# --------------------- Upload splits back to the Hub -------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="NineKnox/tourism-wellness-package",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to the dataset space.")
