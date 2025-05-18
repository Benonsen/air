import pandas as pd
import argparse

def transform_predictions(input_file, output_file, num_predictions=5):
    """
    Transforms a TSV file with space-separated predictions to a TSV file with list-formatted predictions.
    
    Args:
        input_file (str): Path to the input TSV file
        output_file (str): Path to the output TSV file
        num_predictions (int): Number of predictions to keep per post (default: 5)
    """
    # Read the input file
    df = pd.read_csv(input_file, sep='\t')
    
    # Convert space-separated strings to Python lists and limit to num_predictions
    df['preds'] = df['preds'].apply(lambda x: str(x.split()[:num_predictions]))
    
    # Write the output file
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Transformed predictions written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform prediction file format')
    parser.add_argument('--input', type=str, required=True, help='Path to input TSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output TSV file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of predictions to keep (default: 5)')
    
    args = parser.parse_args()
    
    transform_predictions(args.input, args.output, args.top_k)