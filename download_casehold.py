import random
from datasets import load_dataset
import pandas as pd

# Load the casehold/casehold dataset
dataset = load_dataset('casehold/casehold')

# Get the total number of records in the train split
num_records = len(dataset['train'])

# Randomly sample 15000 indices
random_indices = random.sample(range(num_records), 15000)

# Select the data at those indices
selected_data = [dataset['train'][i] for i in random_indices]

# Extract the 'citing_prompt' column and convert to pandas DataFrame
df = pd.DataFrame({'Context': [item['citing_prompt'] for item in selected_data]})

# Save the DataFrame to a CSV file
df.to_csv('data/contexts/train.csv', index=False)

# Repeat the process for the validation split
num_records = len(dataset['validation'])
random_indices = random.sample(range(num_records), 1500)

selected_data = [dataset['validation'][i] for i in random_indices]
df = pd.DataFrame({'Context': [item['citing_prompt'] for item in selected_data]})
df.to_csv('data/contexts/val.csv', index=False)

# Repeat the process for the test split
num_records = len(dataset['test'])
random_indices = random.sample(range(num_records), 1500)

selected_data = [dataset['test'][i] for i in random_indices]
df = pd.DataFrame({'Context': [item['citing_prompt'] for item in selected_data]})
df.to_csv('data/contexts/test.csv', index=False)