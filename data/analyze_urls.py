import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

#python analyze_urls.py processed\PhiUSIIL_Phishing_URL_Dataset.csv raw\combined_urls.csv --limit 200

def find_url_column(df):
    """
    Tries to automatically identify the URL column.
    Returns the column name or None.
    """
    candidates = ['url', 'URL', 'uri', 'URI', 'domain', 'host']
    for col in df.columns:
        if col in candidates:
            return col
        # Fallback: check if column name contains 'url' (case insensitive)
        if 'url' in col.lower():
            return col
    return None

def load_and_measure(filepath, label_name):
    """
    Loads a CSV, finds the URL column, and calculates lengths.
    """
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None

    try:
        # Using on_bad_lines='skip' to be consistent with your loader
        df = pd.read_csv(filepath, on_bad_lines='skip')
        
        url_col = find_url_column(df)
        if not url_col:
            print(f"[Warning] No URL column found in {filepath}. Columns are: {list(df.columns)}")
            return None

        # Calculate length
        # Ensure column is string, handle NaNs
        lengths = df[url_col].astype(str).fillna("").apply(len)
        
        return pd.DataFrame({
            'length': lengths,
            'dataset': label_name
        })

    except Exception as e:
        print(f"[Error] Could not process {filepath}: {e}")
        return None

def print_stats(df_combined, limit=200):
    """
    Prints statistical metrics and the impact of the cutoff limit.
    """
    print("\n" + "="*40)
    print(f"   URL LENGTH STATISTICS")
    print("="*40)
    
    # Group by dataset (File 1 vs File 2)
    stats = df_combined.groupby('dataset')['length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(stats.T)

    print("\n" + "-"*40)
    print(f"   IMPACT OF CUTOFF (Limit: {limit} chars)")
    print("-"*40)
    
    for name, group in df_combined.groupby('dataset'):
        total = len(group)
        # Count how many URLs are strictly longer than the limit
        truncated = len(group[group['length'] > limit])
        percent = (truncated / total) * 100
        print(f"Dataset '{name}':")
        print(f"  - Total URLs: {total}")
        print(f"  - Lost info (truncated): {truncated} ({percent:.2f}%)")
        print(f"  - Max length found: {group['length'].max()}")
        print("")

def plot_distribution(df_combined, limit=200):
    """
    Plots a histogram/KDE of the URL lengths.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_combined, x='length', hue='dataset', kde=True, element="step", bins=100)
    
    # Add the vertical line for the limit used in your CNN
    plt.axvline(limit, color='red', linestyle='--', linewidth=2, label=f'CNN Cutoff ({limit})')
    
    plt.title('URL Length Distribution Comparison')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "url_length_analysis.png"
    plt.savefig(output_file)
    print(f"\n[Info] Plot saved to '{output_file}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze URL lengths from two CSV files.")
    parser.add_argument("file1", type=str, help="Path to first CSV file")
    parser.add_argument("file2", type=str, help="Path to second CSV file")
    parser.add_argument("--limit", type=int, default=200, help="Cutoff limit to simulate (default: 200)")

    args = parser.parse_args()

    # Load data
    data1 = load_and_measure(args.file1, "File 1")
    data2 = load_and_measure(args.file2, "File 2")

    if data1 is not None and data2 is not None:
        # Combine for easier plotting with seaborn
        combined = pd.concat([data1, data2], ignore_index=True)
        
        # Run analysis
        print_stats(combined, args.limit)
        plot_distribution(combined, args.limit)