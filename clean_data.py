import pandas as pd
import html
import re

# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================

df = pd.read_csv("data/amazon_reviews.csv")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. REMOVE UNNECESSARY INDEX COLUMN
# ==========================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ==========================================
# 3. REMOVE DUPLICATE ROWS
# ==========================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("Duplicate rows removed:", before_duplicates - after_duplicates)


# ==========================================
# 4. HANDLE MISSING REVIEWER NAMES
# ==========================================

if "reviewerName" in df.columns:
    df["reviewerName"] = df["reviewerName"].fillna("Unknown")


# ==========================================
# 5. REMOVE REVIEWS WITH MISSING TEXT
# ==========================================

if "reviewText" in df.columns:
    df = df.dropna(subset=["reviewText"])


# ==========================================
# 6. CLEAN REVIEW TEXT
# ==========================================

def clean_text(text):
    text = str(text)

    # Convert HTML entities such as &amp; to &
    text = html.unescape(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at beginning/end
    text = text.strip()

    return text


df["reviewText"] = df["reviewText"].apply(clean_text)


# ==========================================
# 7. CLEAN REVIEWER NAMES
# ==========================================

if "reviewerName" in df.columns:
    df["reviewerName"] = df["reviewerName"].astype(str).str.strip()


# ==========================================
# 8. CHECK RATINGS
# ==========================================

df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

# Remove rows with missing or invalid ratings
df = df.dropna(subset=["overall"])

# Keep only ratings between 1 and 5
df = df[(df["overall"] >= 1) & (df["overall"] <= 5)]


# ==========================================
# 9. RESET INDEX
# ==========================================

df = df.reset_index(drop=True)


# ==========================================
# 10. SAVE CLEANED DATASET
# ==========================================

df.to_csv("data/amazon_reviews_cleaned.csv", index=False)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n========== CLEANING COMPLETED ==========")

print("Cleaned dataset shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nRating distribution:")
print(df["overall"].value_counts().sort_index())

print("\nCleaned dataset saved as:")
print("data/amazon_reviews_cleaned.csv")

print("\nFirst 5 cleaned reviews:")
print(df.head())
print("\n========== FINAL CHECK ==========")

print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nColumns:")
print(df.columns.tolist())