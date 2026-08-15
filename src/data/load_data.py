"""
Load Online Retail II dataset from the raw Excel file.
Combines both sheets into a single DataFrame and performs basic inspection.
"""

import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str = "data/raw/online_retail_II.xlsx") -> pd.DataFrame:
    """
    Load both sheets of the Online Retail II dataset and combine them.
    """
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    
    print("Loading Year 2009-2010 sheet...")
    df1 = pd.read_excel(path, sheet_name="Year 2009-2010")
    
    print("Loading Year 2010-2011 sheet...")
    df2 = pd.read_excel(path, sheet_name="Year 2010-2011")
    
    print("Combining sheets...")
    df = pd.concat([df1, df2], ignore_index=True)
    
    return df


def basic_inspection(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("\n" + "="*60)
    print("BASIC DATA INSPECTION")
    print("="*60)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("\nColumn names and data types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    print(missing_df[missing_df["Missing Count"] > 0])
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic statistics for numerical columns:")
    print(df.describe())


if __name__ == "__main__":
    # Load data
    df = load_raw_data()
    
    # Show basic inspection
    basic_inspection(df)
    
    # Save combined raw data as CSV for easier future use
    output_path = Path("data/raw/online_retail_ii_combined.csv")
    print(f"\nSaving combined data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Saved successfully.")