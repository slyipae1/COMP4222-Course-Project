import pandas as pd

def process_csv(input_file, output_file):
    """
    Processes the input CSV file to separate concatenated answers,
    assigns labels, and saves the output to a new CSV file.

    Parameters:
    - input_file: str, path to the input CSV file
    - output_file: str, path to save the output CSV file
    """
    df = pd.read_csv(input_file)
    new_rows = []

    for index, row in df.iterrows():
        qid = row['qid']
        # Split the answers by newline or any delimiter that separates them
        answers = row['ans'].split('\n')  
        
        # Process each answer
        for i, ans in enumerate(answers):
            ans = ans.strip() 
            if ans:  
                label = 1 if i == 0 else 0 
                new_rows.append((qid, ans, label))  

    new_df = pd.DataFrame(new_rows, columns=['qid', 'ans', 'label'])
    new_df.to_csv(output_file, index=False)

    print(f"Processed CSV file saved to: {output_file}")

if __name__ == "__main__":
    process_csv("result/qwen_data/generated/no_context.csv", "result/qwen_data/generated/no_context_encoded.csv")
    process_csv("result/qwen_data/generated/with_context.csv", "result/qwen_data/generated/with_context_encoded.csv")