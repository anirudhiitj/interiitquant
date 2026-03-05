#!/usr/bin/env python3
import sys
from pathlib import Path
import dask.dataframe as dd

def convert_csv_to_parquet(csv_path):
    parquet_path = csv_path.with_suffix('.parquet')
    print(f"Converting {csv_path.name}...", end=' ')
    
    ddf = dd.read_csv(csv_path)
    ddf.to_parquet(parquet_path, engine='pyarrow', write_index=False)
    
    csv_path.unlink()
    
    print(f"✓ Created {parquet_path.name} (original deleted)")

def convert_directory(input_dir):
    input_dir = Path(input_dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid directory: {input_dir}")
    
    csv_files = list(input_dir.glob("**/day*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to convert\n")
    
    converted = 0
    for csv_file in csv_files:
        try:
            convert_csv_to_parquet(csv_file)
            converted += 1
        except Exception as e:
            print(f"✗ Error converting {csv_file.name}: {e}")
    
    print(f"\nConverted {converted}/{len(csv_files)} files successfully")

if __name__ == "__main__":
    INPUT_DIRECTORY = "/data/quant14/"  # Set your directory path here
    
    try:
        convert_directory(INPUT_DIRECTORY)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)