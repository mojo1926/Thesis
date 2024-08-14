import pandas as pd
import numpy as np
import re
import os
from openai import OpenAI

# Define file paths
sample_file_path = 'Data/All_Service_Risks.xlsx'
test_dataset_path = 'Data/test_service_risks.xlsx'
remaining_content_path = 'Data/companies_stock_data.xlsx'
output_file_path = 'C:/Users/antho/OneDrive/Masterthesis/Thesis/Risks/Risk_Comparison.xlsx'

client = OpenAI(api_key=os.getenv('api_key_gpt'))

# Load risks from an Excel file and sample a few examples
def load_and_sample_risks(file_path, sample_size=5):
    df = pd.read_excel(file_path)
    return df

def get_filing_without_item1a(ticker, df_remaining):
    row = df_remaining[df_remaining['ticker'] == ticker]
    if row.empty:
        print(f"No data found for ticker {ticker}.")
        return ""
    remaining_content = row['remaining_content'].values[0]
    cleaned_content = clean_content(remaining_content)
    return cleaned_content

def clean_content(content):
    lines = content.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip() and re.search("[a-zA-Z0-9]", line)]
    cleaned_content = "\n".join(cleaned_lines)
    return cleaned_content

# Function to make the API request to ChatGPT
def request_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0,
        max_tokens=4096,
        messages=[
                    {"role": "system", "content": "You are a helpful financial risk analyst."},
                    {"role": "user", "content":f"{prompt}"}
                ]
    )
    return response.choices[0].message.content.strip().split('\n')

# Generate the prompt
def generate_prompt(ticker, sampled_risks, industry_name, df_remaining):
    filing_without_item1a = get_filing_without_item1a(ticker, df_remaining)
    risks_text = '\n'.join([f"- {risk}" for risk in sampled_risks])
    prompt = f"""
    You review the 10-K filings information provided and your own knowledge base to identify financial and business risk factors for the company.
    Generate risk factors for a company in the {industry_name} industry. The risk factors should be based on the information provided in the following text which is a SEC-EDGAR 10K filing.
    {filing_without_item1a}
    Create a set of facts for the provided text. This set of facts is separate from the risks and should not be included in the later output.
    To create a risk factor, analyze the provided information. Then define possible risks based on this information. Specify risks if they are related to a specific business process or department.
    Use related information you have about the service sector from outside of this filing content for the filing period 2021-2022 and combine it with the given information.
    Here are some few shot examples of how the risks factors should look like from a {industry_name} company risk pool:\n{risks_text}\n\n
    Use any information you have about the company and its industry in the United States except information that is from the year 2023 to generate the risks.
    Now check if the set of facts you created earlier is consistent with the risk factors you generated. If not, adjust the risks that are inconsistent.
    Do not add anything else to the output prompt other than the risk factors.
    """
    return prompt

# Step 1: Pull list with all service companies and their actual risks
def create_filed_risks_sheet(test_dataset_path, output_file_path):
    df = pd.read_excel(test_dataset_path)
    filed_risks = df[['Ticker', 'Risks']]
    
    with pd.ExcelWriter(output_file_path, mode='w') as writer:
        filed_risks.to_excel(writer, sheet_name='Filed Risks', index=False)

# Step 2-4: For each ticker
def process_tickers(test_dataset_path, sample_file_path, remaining_content_path, output_file_path):
    df = pd.read_excel(test_dataset_path)
    df_remaining = pd.read_excel(remaining_content_path)
    tickers = df['Ticker'].unique()
    avg_risks_per_company = 10

    # Check existing sheets to avoid processing the same ticker twice
    try:
        existing_sheets = pd.ExcelFile(output_file_path).sheet_names
    except FileNotFoundError:
        existing_sheets = []

    prompt_sample_risks = []
    random_risks = []
    identified_risks = []

    full_risk_df = load_and_sample_risks(sample_file_path)
    
    for ticker in tickers:
        if f'{ticker} Prompt Sample Risks' in existing_sheets or f'{ticker} Identified Risks' in existing_sheets:
            print(f"Ticker {ticker} already processed. Skipping.")
            continue

        print(f"Processing ticker: {ticker}")
        sampled_risks = full_risk_df['Risks'].sample(n=5).tolist()
        prompt = generate_prompt(ticker, sampled_risks, "Service", df_remaining)
        identified_risks_list = request_chatgpt(prompt)

        # Save risks into lists
        prompt_sample_risks.append({'Ticker': ticker, 'Sampled Risks': sampled_risks})
        identified_risks.append({'Ticker': ticker, 'Identified Risks': identified_risks_list})
        random_risks.append({'Ticker': ticker, 'Random Risks': full_risk_df['Risks'].sample(n=avg_risks_per_company).tolist()})

    # Write all data to Excel
    with pd.ExcelWriter(output_file_path, mode='a', if_sheet_exists='overlay') as writer:
        if 'Prompt Sample Risks' not in existing_sheets:
            prompt_sample_risks_df = pd.DataFrame(prompt_sample_risks).explode('Sampled Risks')
            prompt_sample_risks_df.to_excel(writer, sheet_name='Prompt Sample Risks', index=False)
        if 'Identified Risks' not in existing_sheets:
            identified_risks_df = pd.DataFrame(identified_risks).explode('Identified Risks')
            identified_risks_df.to_excel(writer, sheet_name='Identified Risks', index=False)
        if 'Random Risks' not in existing_sheets:
            random_risks_df = pd.DataFrame(random_risks).explode('Random Risks')
            random_risks_df.to_excel(writer, sheet_name='Random Risks', index=False)

# Run the steps
create_filed_risks_sheet(test_dataset_path, output_file_path)
process_tickers(test_dataset_path, sample_file_path, remaining_content_path, output_file_path)

print("Process completed.")
