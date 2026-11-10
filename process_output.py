import os
import json
import pandas as pd

def find_json_files(directory):
    """Find all .json files in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

def extract_data_from_json(file_path):
    """Extract relevant data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def combine_json_data(files):
    """Combine data from multiple JSON files into a single dataframe."""
    all_data = []
    for file in files:
        json_data = extract_data_from_json(file)
        all_data.append(json_data)
    return pd.DataFrame(all_data)

def export_to_excel(df, output_file):
    """Export the dataframe to an Excel file."""
    df.to_excel(output_file, index=False)

# Main function to run the process
def main(directory, output_file):
    json_files = find_json_files(directory)
    combined_data = combine_json_data(json_files)
    export_to_excel(combined_data, output_file)
    print(f"Data exported successfully to {output_file}")

# Usage example:
# Replace 'your_directory' with the folder path and 'output.xlsx' with the desired output filename.
if __name__ == "__main__":
    main('.', 'output.xlsx')
