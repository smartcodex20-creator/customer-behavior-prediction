"""
Deep Data Validation for Online Retail II dataset.
Performs schema checks, range checks, and quality reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_interim_data(path: str = "data/interim/customer_transactions.csv") -> pd.DataFrame:
    """Load the cleaned customer-level data."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Check expected columns and data types."""
    expected_columns = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer_ID", "Country",
        "Is_Return", "Invalid_Price"
    ]
    
    results = {
        "missing_columns": [col for col in expected_columns if col not in df.columns],
        "extra_columns": [col for col in df.columns if col not in expected_columns],
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return results


def validate_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Check logical ranges and data quality issues."""
    results = {
        "total_rows": len(df),
        "date_range": {
            "min": str(df["InvoiceDate"].min()),
            "max": str(df["InvoiceDate"].max())
        },
        "quantity": {
            "min": int(df["Quantity"].min()),
            "max": int(df["Quantity"].max()),
            "negative_count": int((df["Quantity"] < 0).sum()),
            "zero_count": int((df["Quantity"] == 0).sum())
        },
        "price": {
            "min": float(df["Price"].min()),
            "max": float(df["Price"].max()),
            "zero_or_negative": int((df["Price"] <= 0).sum())
        },
        "customer_id": {
            "unique_customers": int(df["Customer_ID"].nunique()),
            "missing": int(df["Customer_ID"].isna().sum())
        },
        "invoice": {
            "unique_invoices": int(df["Invoice"].nunique())
        },
        "country": {
            "unique_countries": int(df["Country"].nunique()),
            "top_5": df["Country"].value_counts().head(5).to_dict()
        }
    }
    return results


def print_validation_report(schema_results: Dict, range_results: Dict) -> None:
    """Pretty print the validation results."""
    print("\n" + "="*70)
    print("DEEP DATA VALIDATION REPORT")
    print("="*70)

    print("\n1. SCHEMA CHECK")
    print("-" * 40)
    if schema_results["missing_columns"]:
        print(f"Missing columns: {schema_results['missing_columns']}")
    else:
        print("All expected columns are present.")
    
    if schema_results["extra_columns"]:
        print(f"Extra columns: {schema_results['extra_columns']}")
    
    print("\nData Types:")
    for col, dtype in schema_results["dtypes"].items():
        print(f"  {col:<20} → {dtype}")

    print("\n2. RANGE & QUALITY CHECK")
    print("-" * 40)
    print(f"Total rows                : {range_results['total_rows']:,}")
    print(f"Date range                : {range_results['date_range']['min']} → {range_results['date_range']['max']}")
    print(f"Unique customers          : {range_results['customer_id']['unique_customers']:,}")
    print(f"Unique invoices           : {range_results['invoice']['unique_invoices']:,}")
    print(f"Unique countries          : {range_results['country']['unique_countries']}")
    
    print(f"\nQuantity:")
    print(f"  Min / Max               : {range_results['quantity']['min']} / {range_results['quantity']['max']}")
    print(f"  Negative (Returns)      : {range_results['quantity']['negative_count']:,}")
    print(f"  Zero quantity           : {range_results['quantity']['zero_count']:,}")
    
    print(f"\nPrice:")
    print(f"  Min / Max               : {range_results['price']['min']} / {range_results['price']['max']}")
    print(f"  Zero or Negative        : {range_results['price']['zero_or_negative']:,}")

    print(f"\nTop 5 Countries:")
    for country, count in range_results["country"]["top_5"].items():
        print(f"  {country:<25} : {count:,}")


if __name__ == "__main__":
    df = load_interim_data()
    
    schema_results = validate_schema(df)
    range_results = validate_ranges(df)
    
    print_validation_report(schema_results, range_results)
    
    print("\nValidation completed.")