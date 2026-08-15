"""
Data cleaning module for Online Retail II dataset.
Follows the requirements in the Product Requirements Document (Section 6).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_combined_data(path: str = "data/raw/online_retail_ii_combined.csv") -> pd.DataFrame:
    """Load the combined raw CSV."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform cleaning according to PRD requirements.
    
    Returns
    -------
    customer_df : pd.DataFrame
        Rows with valid Customer ID (for customer-level modeling)
    product_df : pd.DataFrame
        All rows (useful for product-level analysis)
    """
    print("\nStarting data cleaning...")
    original_rows = len(df)
    
    # 1. Standardize column names (remove spaces)
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    
    # 2. Convert Customer ID to nullable integer
    df["Customer_ID"] = pd.to_numeric(df["Customer_ID"], errors="coerce").astype("Int64")
    
    # 3. Flag returns (negative quantity)
    df["Is_Return"] = df["Quantity"] < 0
    
    # 4. Handle zero or negative prices (keep for now but flag)
    df["Invalid_Price"] = df["Price"] <= 0
    
    # 5. Remove exact duplicate rows (keep legitimate repeated purchases)
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before_dedup - len(df):,} exact duplicate rows")
    
    # 6. Separate customer-level data (must have Customer ID)
    customer_df = df[df["Customer_ID"].notna()].copy()
    product_df = df.copy()  # keep everything for product analysis
    
    print(f"\nOriginal rows          : {original_rows:,}")
    print(f"After cleaning         : {len(df):,}")
    print(f"Rows with Customer ID  : {len(customer_df):,}")
    print(f"Rows without Customer ID: {len(df) - len(customer_df):,}")
    print(f"Return transactions    : {df['Is_Return'].sum():,}")
    print(f"Invalid price rows     : {df['Invalid_Price'].sum():,}")
    
    return customer_df, product_df


def save_cleaned_data(customer_df: pd.DataFrame, product_df: pd.DataFrame) -> None:
    """Save cleaned datasets to data/interim/"""
    interim_path = Path("data/interim")
    interim_path.mkdir(parents=True, exist_ok=True)
    
    customer_path = interim_path / "customer_transactions.csv"
    product_path = interim_path / "all_transactions.csv"
    
    customer_df.to_csv(customer_path, index=False)
    product_df.to_csv(product_path, index=False)
    
    print(f"\nSaved customer-level data → {customer_path}")
    print(f"Saved full transaction data → {product_path}")


if __name__ == "__main__":
    # Load
    df = load_combined_data()
    
    # Clean
    customer_df, product_df = clean_data(df)
    
    # Save
    save_cleaned_data(customer_df, product_df)
    
    print("\nCleaning completed successfully.")