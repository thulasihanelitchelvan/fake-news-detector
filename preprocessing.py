import pandas as pd
import re
import string
from sklearn.utils import shuffle
from pathlib import Path

# Get the current script directory
current_dir = Path(__file__).parent

# Move up to the project root
project_root = current_dir

# Target files inside data folder
fake_path = project_root / "data" / "fake.csv"
real_path = project_root / "data" / "real.csv"
output_path = project_root / "data" / "cleaned_data.csv"

fake = pd.read_csv(fake_path, encoding="utf-8")
real = pd.read_csv(real_path, encoding="utf-8")

# ---------- Text cleaning ----------
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)                      # remove links
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)                          # remove numbers
    text = re.sub(r"\s+", " ", text).strip()                 # normalize spaces
    return text

def _best_text_column(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns]
    has_text = "text" in cols
    has_title = "title" in cols

    if has_text and has_title:
        merged = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
        return merged
    if has_text:
        return df["text"].fillna("").astype(str)
    if has_title:
        return df["title"].fillna("").astype(str)
    raise KeyError("CSV must contain at least a 'text' or 'title' column.")

def preprocess():
    # Load datasets (robust to encodings)
    fake = pd.read_csv(fake_path, encoding="utf-8",)
    real = pd.read_csv(real_path, encoding="utf-8",)

    # Labels: 0 = FAKE, 1 = REAL (keep consistent across pipeline)
    fake["label"] = 0
    real["label"] = 1

    # Select/compose text column
    fake_text = _best_text_column(fake)
    real_text = _best_text_column(real)
    fake = pd.DataFrame({"text": fake_text, "label": fake["label"]})
    real = pd.DataFrame({"text": real_text, "label": real["label"]})

    # Combine
    df = pd.concat([fake, real], ignore_index=True)

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Drop empty + duplicates
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset="text")

    # Shuffle
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Cleaned data saved to {output_path}")
    print(df["label"].value_counts().rename({0: "FAKE(0)", 1: "REAL(1)"}))

if __name__ == "__main__":
    preprocess()
