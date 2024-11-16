from datasets import load_dataset
import json

# Load the dataset
ds = load_dataset("allenai/sciq", split='train')
# Shuffle the dataset (optional, if random selection is desired)
ds = ds.shuffle(seed=42)

# Select the first 2000 samples
subset = ds.select(range(2000))

# Prepare the data
formatted_data = []

for idx, item in enumerate(subset):
    # Create the required structure
    formatted_item = {
        "data": {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "question": item['question'],
                            "id": str(idx),  # Use the index as a unique ID
                            "answers": [
                                {
                                    "text": item['correct_answer'],
                                    "answer_start": 0
                                }
                            ]
                        }
                    ],
                    "context": item['support']
                }
            ]
        }
    }
    
    # Append to the list
    formatted_data.append(formatted_item)

# Write to a JSON file
with open('..result/SciQ_result/data/SciQdata_2000.json', 'w') as f:
    for entry in formatted_data:
        json.dump(entry, f)
        f.write('\n')