import os
import pandas as pd
import re
import time
from bs4 import BeautifulSoup
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Load the .env file to get the API key
load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv('api_key_gpt'))

# Function to analyze risks in the text
def analyze_risks_in_file(text):

    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response =  client.chat.completions.create(
                temperature=0,
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful financial risk analyst."},
                    {"role": "user", "content":
                        f"""
                        Please analyze the following text to identify distinct risks mentioned in it. Start your analysis from the section titled 'Item 1A. Risk Factors.' For each identified risk:

                        1. Extract the risk as a separate entry.
                        2. Keep the original text intact.
                        3. If a risk is mentioned as part of a list, rephrase it as a standalone sentence while retaining its original meaning.

                        Every sentence or paragraph in this text states one or more risks. Identify each risk by its inherent meaning and context in the paragraph and format as follows:
                        [paragraph1 risk1]
                        [paragraph1 risk2]
                        [paragraph2 risk1]
                        [paragraph3 risk1]
                        ...

                        {text}

                        Note: The text may contain complex and lengthy paragraphs or lists of risks of a single commodity or instance ('[commodity or instance]: [risk1]; [risk2]; [risk3].').
                        Check these lists and separate the different risks as follows:
                        [commodity or instance or other text][risk1]
                        [commodity or instance or other text][risk2]
                        ...

                        Please reply only with the identified risks, formatted as follows:
                        [Risk 1]
                        [Risk 2]
                        ...

                        Please ensure each risk is captured accurately. Don't put the risks in '[]' or 'Risk ...:' in front of it.
                        """
                    }
                ]
            )
            return response.choices[0].message.content.strip().split('\n')
        except OpenAI.error.InternalServerError as e:
            if attempt < retry_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e

# Function to process the cleaned data and extract risks
def process_data(file_path, output_file_path):
    data = []

    # Load the cleaned Excel file
    df = pd.read_excel(file_path, sheet_name='service')

 # Check if the output file exists and is a valid Excel file
    if os.path.exists(output_file_path):
        try:
            existing_df = pd.read_excel(output_file_path, sheet_name='service')
            existing_data = existing_df.to_dict('records')
            data.extend(existing_data)
        except Exception as e:
            print(f"Error reading {output_file_path}: {e}")
            print("Starting with an empty dataset.")
    else:
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            writer.save()

    processed_companies = {entry['Name']: entry for entry in data}

    for _, row in df.iterrows():
        if row['name'] in processed_companies:
            continue

        print(f"Starting risk extraction for: {row['name']}")

        item1a_content = row['item1a_content']

        if pd.notna(item1a_content):
            risks = analyze_risks_in_file(item1a_content)

            for risk in risks:
                data.append({
                    'Name': row['name'],
                    'Ticker': row['ticker'],
                    'Risks': risk,
                    'Sentiment': row['Sentiment']
                })

        # Save progress to the output file
        pd.DataFrame(data).to_excel(output_file_path, sheet_name='service', index=False)

        print(f"Completed risk extraction for: {row['name']}")

    return pd.DataFrame(data)

# Process the cleaned data and extract risks
cleaned_file_path = 'Data/train_service_companies_stock_data.xlsx'
output_file_path = 'Data/extracted_train_risks_service_companies.xlsx'
risk_data_df = process_data(cleaned_file_path, output_file_path)

print(f"Extracted risks have been saved to {output_file_path}")