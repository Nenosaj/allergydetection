import pandas as pd
import numpy as np

def clean_csv(file_path, output_path, delimiter=',', encoding='utf-8'):
    """
    Cleans inconsistent CSV data and saves the cleaned version.
    
    Parameters:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.
        delimiter (str): Delimiter used in the CSV file (default is comma).
        encoding (str): Encoding of the CSV file (default is UTF-8).
    """
    try:
        # Step 1: Load the file
        print("Loading file...")
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, on_bad_lines='skip')
        print("File loaded successfully.")
        
        # Step 2: Standardize column names
        print("Standardizing column names...")
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Step 3: Handle missing values
        print("Handling missing values...")
        missing_threshold = 0.5  # Drop columns with >50% missing values
        df = df.dropna(axis=1, thresh=int(missing_threshold * len(df)))  # Drop columns
        df = df.fillna('Unknown')  # Replace remaining NaNs with 'Unknown'

        # Step 4: Trim spaces
        print("Trimming extra spaces...")
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Step 5: Remove duplicate rows
        print("Removing duplicate rows...")
        df = df.drop_duplicates()

        # Step 6: Convert data types
        print("Converting data types...")
        for column in df.select_dtypes(include=['object']):
            # Attempt to convert to numeric or datetime, fallback to string
            try:
                df[column] = pd.to_numeric(df[column], errors='ignore')
                df[column] = pd.to_datetime(df[column], errors='ignore')
            except Exception as e:
                print(f"Could not convert column '{column}': {e}")

        # Step 7: Handle invisible or non-printable characters
        print("Removing invisible characters...")
        df = df.applymap(lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else x)

        # Step 8: Save the cleaned file
        print(f"Saving cleaned data to {output_path}...")
        df.to_csv(output_path, index=False)
        print("Cleaned data saved successfully.")
        
    except Exception as e:
        print(f"Error while processing: {e}")


# Example usage
file_path = 'dataset/13k-recipes.csv'  # Replace with the path to your input CSV file
output_path = 'cleaned_output.csv'  # Replace with the path to save the cleaned CSV
clean_csv(file_path, output_path)
