import pandas as pd

file_names = ['data/lfqa/test.csv', 'data/lfqa/train.csv', 'data/lfqa/val.csv']

tokens = {
    'context_token': '<context>',
    'question_token': '<question>',
    'issues_token': '<issues>',
    'rules_token': '<rules>',
    'analysis_token': '<analysis>',
    'answer_token': '<answer>',
    'end_token': '<end>'
}


for f in file_names:    
    # Load the dataset
    df = pd.read_csv(f)

    # Create a single 'text' column
    df['text'] = tokens['context_token'] + ' ' + df['Context'] + ' ' + \
                    tokens['question_token'] + ' ' + df['Question'] + ' ' + \
                    tokens['answer_token'] + ' ' + df['Answer'] + ' ' + \
                    tokens['end_token']
    
    # Save the processed dataset
    df[['text']].to_csv(f.replace('.csv', '_processed.csv'), index=False)

    df['text'] = tokens['context_token'] + ' ' + df['Context'] + ' ' + \
             tokens['question_token'] + ' ' + df['Question'] + ' ' + \
             tokens['issues_token'] + ' ' + df['Issues'] + ' ' + \
             tokens['rules_token'] + ' ' + df['Rules'] + ' ' + \
             tokens['analysis_token'] + ' ' + df['Analysis'] + ' ' + \
             tokens['answer_token'] + ' ' + df['Answer'] + ' ' + \
             tokens['end_token']
    
    # Save the processed dataset
    df[['text']].to_csv(f.replace('.csv', '_processed_with_r.csv'), index=False)



