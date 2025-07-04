STRICT_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers"
]

EXPANDED_PRODUCTS = STRICT_PRODUCTS + [
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Money transfer, virtual currency, or money service",
]
import pandas as pd
import re

# Define product categories for different filtering modes



def load_data(path: str) -> pd.DataFrame:
    """Load complaint dataset from CSV."""
    return pd.read_csv(path)

def filter_products(df: pd.DataFrame, mode: str = "strict") -> pd.DataFrame:
    """
    Filter products according to the selected mode:
    - strict: only core 5 products
    - expanded: strict + adjacent relevant products
    - all: no filtering on Product (except removing empty narratives)
    """
    if mode == "strict":
        products = STRICT_PRODUCTS
    elif mode == "expanded":
        products = EXPANDED_PRODUCTS
    elif mode == "all":
        # No filtering by product
        return df
    else:
        raise ValueError(f"Unknown filter mode: {mode}")

    return df[df["Product"].isin(products)]

def remove_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without complaint narratives."""
    return df[df["Consumer complaint narrative"].notna()]

def clean_text(text: str) -> str:
    """Clean individual text narrative."""
    text = str(text).lower()
    text = re.sub(r"\b(i am writing to file a complaint|i would like to report)\b", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to all complaint narratives."""
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)
    return df

def remove_short_narratives(df: pd.DataFrame, min_words: int = 10) -> pd.DataFrame:
    """Remove complaints with very short cleaned narratives."""
    df["word_count"] = df["cleaned_narrative"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= min_words]
    df.drop(columns=["word_count"], inplace=True)
    return df

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save preprocessed data to CSV."""
    df.to_csv(output_path, index=False)

def preprocess_complaints(
    df: pd.DataFrame,
    apply_min_word_filter: bool = False,
    min_words: int = 5,
    filter_mode: str = "strict"
) -> pd.DataFrame:
    """
    Full preprocessing pipeline with optional filtering mode and length filter.
    
    - Filters products according to filter_mode
    - Removes empty narratives
    - Cleans narratives
    - Optionally removes short narratives
    """
    df = filter_products(df, mode=filter_mode)
    df = remove_empty_narratives(df)
    df = clean_narratives(df)

    if apply_min_word_filter:
        df = remove_short_narratives(df, min_words)

    return df
