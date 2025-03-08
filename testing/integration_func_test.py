import pandas as pd
import pytest

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame by cleaning column names, handling duplicates,
    converting dates, and filling missing values.
    """
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Drop duplicates
    df = df.drop_duplicates()

    # Convert date column to datetime
    if 'date_column' in df.columns:
        df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

    # Fill missing values in a numeric column with the median
    if 'numeric_column' in df.columns:
        median_value = df['numeric_column'].median()
        df['numeric_column'].fillna(median_value, inplace=True)

    return df

@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame."""
    data = {
        "Name": ["Alice", "Bob", "Charlie", "Alice"],
        "Date Column": ["2023-01-01", "2023-02-01", "InvalidDate", "2023-01-01"],
        "Numeric Column": [10, None, 30, 10],
    }
    return pd.DataFrame(data)

def test_process_dataframe(sample_dataframe):
    """Tests the process_dataframe function."""
    processed_df = process_dataframe(sample_dataframe)

    # Validate column name transformation
    assert list(processed_df.columns) == ["name", "date_column", "numeric_column"]

    # Validate duplicate removal (Alice's row should be removed)
    assert processed_df.shape[0] == 3  # Originally 4, now 3

    # Validate date parsing (invalid date should be NaT)
    assert pd.isna(processed_df.loc[processed_df["name"] == "charlie", "date_column"]).all()

    # Validate missing numeric value filled with median
    median_value = sample_dataframe["Numeric Column"].median()
    assert processed_df["numeric_column"].isna().sum() == 0  # No missing values
    assert processed_df.loc[processed_df["name"] == "bob", "numeric_column"].values[0] == median_value