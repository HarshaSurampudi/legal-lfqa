import openai
import pandas as pd
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Initialize the OpenAI API with your key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv('data/contexts/train.csv')
test_df = pd.read_csv('data/contexts/test.csv')
val_df = pd.read_csv('data/contexts/val.csv')

def parse_response(text):
    """Parse the text to json and return the json."""
    try:
        json_obj =  json.loads(text)
        return json_obj
    except:
        print("Error: Invalid json format.")
        return None

response_schema = """{
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "A question that can be answered based on the given context. Asked by non legal expert to the legal expert."
        }
        "intermediate_steps": {
            "type": "object",
            "description": "Intermediate steps that were taken to reach the final answer, following the IRAC method.",
            "properties": {
                "Issues": {
                    "type": "string",
                    "description": "Issues that were identified in the question."
                },
                "Rules": {
                    "type": "string",
                    "description": "Rules that are applicable to the identified issues."
                },
                "Analysis": {
                    "type": "string",
                    "description": "Analysis of the identified issues based on the applicable rules."
                }
            }
        },
        "answer": {
            "type": "string",
            "description": "Answer to the question based on the given context and the intermediate steps."
            }
        }
    }"""

def get_response(system_prompt, user_prompt):
    """Get the response from the OpenAI API."""
    response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ])
    
    message = response["choices"][0]["message"]["content"]
    return message 

def generate_lfqa_record(context):
    system_prompt = "Your task is to generate one synthetic dataset record in json format following given instructions and schema."
    user_prompt = """Given the following legal context passage, generate a realistic synthetic dataset record for the task of Long Form Question Answering using legal reasoning, following the provided schema.
    Context: {}
    Schema: {}
    """.format(context, response_schema)
    message = get_response(system_prompt, user_prompt)
    parsed_message = parse_response(message)
    record = {}
    record["question"] = parsed_message["question"]
    record["Issues"] = parsed_message["intermediate_steps"]["Issues"]
    record["Rules"] = parsed_message["intermediate_steps"]["Rules"]
    record["Analysis"] = parsed_message["intermediate_steps"]["Analysis"]
    record["answer"] = parsed_message["answer"]
    return record

def generate_lfqa_dataset(df):
    """Generate the dataset for the task of Long Form Question Answering using legal reasoning."""
    dataset = []
    for index, row in df.iterrows():
        context = row["Context"]
        record = generate_lfqa_record(context)
        record["context"] = context
        dataset.append(record)
    dataset_df = pd.DataFrame(dataset)
    return dataset_df
