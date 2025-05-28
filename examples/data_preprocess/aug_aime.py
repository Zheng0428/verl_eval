from pandas import read_csv, read_parquet
import pandas as pd
import os


def augment_aime_data(input_path, output_path=None):
    """
    Read a parquet file, filter out AIME data, duplicate it 8 times,
    merge it back into the original data, and save as parquet.
    
    Args:
        input_path: Path to the input parquet file
        output_path: Path to save the output parquet file. If None, will use input_path with '_augmented' suffix
    """
    # Determine output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_augmented{ext}"
    
    # Read the parquet file
    print(f"Reading data from {input_path}")
    df = read_parquet(input_path)
    original_count = len(df)
    print(f"Original data contains {original_count} rows")
    
    # Filter out the AIME data
    aime_data = df[df["data_source"] == "simplelr_aime24"].copy()
    aime_count = len(aime_data)
    print(f"Found {aime_count} rows with data_source='simplelr_aime24'")
    
    # Duplicate the AIME data 8 times
    augmented_data = pd.concat([aime_data] * 8, ignore_index=True)
    
    # Modify the data_source field
    augmented_data["data_source"] = "simplelr_aime24_avg8"
    print(f"Created {len(augmented_data)} augmented rows")
    
    # Merge the augmented data back into the original data
    merged_data = pd.concat([df, augmented_data], ignore_index=True)
    print(f"Merged data contains {len(merged_data)} rows (original: {original_count}, new: {len(augmented_data)})")
    
    # Save as parquet
    merged_data.to_parquet(output_path, index=False)
    print(f"Saved augmented data to {output_path}")
    
    return merged_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment AIME data in a parquet file")
    parser.add_argument("input_path", help="Path to the input parquet file")
    parser.add_argument("--output_path", "-o", help="Path to save the output parquet file")
    
    args = parser.parse_args()
    augment_aime_data(args.input_path, args.output_path)

