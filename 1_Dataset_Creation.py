import os
import pandas as pd
import re
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import openpyxl
from sec_edgar_downloader import Downloader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
excel_file_path = r'C:\Users\antho\OneDrive\Masterthesis\Thesis\Data\Yahoo_All_Sectors_Ticker.xlsx'
xls = pd.ExcelFile(excel_file_path)

# Dictionary to hold companies data by industry
industry_companies = {}

# Iterate through each sheet (industry)
for sheet_name in xls.sheet_names:
    sector_data = pd.read_excel(xls, sheet_name=sheet_name)

    # Ensure a list exists for the industry
    industry_key = sheet_name.replace(" ", "_").lower()
    if industry_key not in industry_companies:
        industry_companies[industry_key] = []

    # Process each row in the sheet
    for _, row in sector_data.iterrows():
        # Create a dictionary with all available data
        company = {
            'name': row.get('Name'),
            'ticker': row.get('Ticker'),
            'industry': sheet_name,
            'yahoo_ticker': row.get('Ticker'),
            'filing_date': None,
            'sic': None,
            'sic_description': None,
            'state': None,
            'fiscal_year_end': None,
            'stock_prices': {"At Filing Date": None, "1 Year After": None},
            'has_item1a': False,
            'item1a_content': '',
            'remaining_content': ''
        }
        # Add the company to the current industry's list
        industry_companies[industry_key].append(company)

def initialize_downloader(industry_name):
    formatted_industry_name = industry_name.replace(" ", "_")  # Replace spaces with underscores
    file_saving_path = f"C:/Users/antho/OneDrive/Masterthesis/Thesis/10K_Sample_Files/{formatted_industry_name}"
    dl = Downloader("University of Kassel", "anthony.borgan@wi-kassel.de", file_saving_path)
    return dl

def extract_filing_date_and_details(content):
    filing_date = None
    sic = None
    sic_description = None
    state = None
    fiscal_year_end = None

    for line in content.split('\n')[:30]:
        if 'FILED AS OF DATE:' in line:
            match_date = re.search(r'FILED AS OF DATE:\s+(\d{4})(\d{2})(\d{2})', line)
            if match_date:
                filing_date = f"{match_date.group(1)}-{match_date.group(2)}-{match_date.group(3)}"

        if 'STANDARD INDUSTRIAL CLASSIFICATION:' in line:
            match_sic = re.search(r'STANDARD INDUSTRIAL CLASSIFICATION:\s+(.*?)\s+\[(\d+)\]', line)
            if match_sic:
                sic_description, sic = match_sic.group(1).strip(), match_sic.group(2)

        if 'STATE OF INCORPORATION:' in line:
            match_state = re.search(r'STATE OF INCORPORATION:\s+(\w{2})', line)
            if match_state:
                state = match_state.group(1)

        if 'FISCAL YEAR END:' in line:
            match_fye = re.search(r'FISCAL YEAR END:\s+(\d{4})', line)
            if match_fye:
                fiscal_year_end = match_fye.group(1)

    return filing_date, sic, sic_description, state, fiscal_year_end

def find_txt_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                return os.path.join(root, file)
    return None

def download_and_extract_item1a(company, dl):
    try:
        identifier = company['ticker'] if company['ticker'] else company['cik']
        print(identifier)
        if identifier:
            dl.get(form="10-K", ticker_or_cik=identifier, after="2021-12-31", before="2022-12-31")
            print(f"Files for {identifier} successfully downloaded.")
            download_path = Path(f"C:/Users/antho/OneDrive/Masterthesis/Thesis/10K_Sample_Files/service/sec-edgar-filings/{identifier}")
            print(f"download_path: {download_path}")

            try:
                file_path = find_txt_file(download_path)
                if file_path:
                    print(f"file path for content of {identifier}: {file_path}")
                    with open(file_path, 'r') as file:
                        content = file.read()
                        item1a_content, remaining_content, has_item1a = extract_item1a(content)
                        filing_date, sic, sic_description, state, fiscal_year_end = extract_filing_date_and_details(content)

                        print(f"Filing date: {filing_date}, SIC: {sic}, SIC Description: {sic_description}, State: {state}, Fiscal Year End: {fiscal_year_end}")

                        company['has_item1a'] = has_item1a
                        company['item1a_content'] = item1a_content
                        company['remaining_content'] = remaining_content
                        company['filing_date'] = filing_date
                        company['sic'] = sic
                        company['sic_description'] = sic_description
                        company['state'] = state
                        company['fiscal_year_end'] = fiscal_year_end
                else:
                    print("no file path found")
            except Exception as e:
                print(f"Error while processing file: {e}")

    except Exception as e:
        print(f"Failed to download for {company['name']} ({identifier}): {e}")

def extract_item1a(content):
    # Initialize document dictionary to store 10-K content
    document = {}

    # Regex patterns for document start, end, and type
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    # Finding all occurrences of the patterns
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(content)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(content)]
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(content)]

    # Extracting 10-K section
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type.strip() == '10-K':
            document['10-K'] = content[doc_start:doc_end]

    if '10-K' not in document:
        return "", content, False  # If no 10-K section is found, return empty Item 1A and original content as remaining

    # Regex to find Item 1A and the sections that follow it
    item_1a_start_pattern = re.compile(r'ITEM 1A\.\s*RISK FACTORS', re.IGNORECASE)

    # Attempt to find ITEM 1A section start
    match = item_1a_start_pattern.search(document['10-K'])
    if not match:
        return "", document['10-K'], True  # If ITEM 1A not found, return empty Item 1A and 10-K content as remaining

    start_of_item_1a = match.start()

    # Assuming ITEM 1B as the next section to determine the end of ITEM 1A
    item_1b_start_pattern = re.compile(r'ITEM 1B\.', re.IGNORECASE)
    end_match = item_1b_start_pattern.search(document['10-K'], start_of_item_1a)

    end_of_item_1a = end_match.start() if end_match else len(document['10-K'])

    # Extracting ITEM 1A content
    item_1a_content = document['10-K'][start_of_item_1a:end_of_item_1a]

    # Remaining content
    remaining_content = document['10-K'][:start_of_item_1a] + document['10-K'][end_of_item_1a:]

    # Convert extracted content to text
    item_1a_soup = BeautifulSoup(item_1a_content, 'lxml')
    remaining_soup = BeautifulSoup(remaining_content, 'lxml')

    item_1a_text = item_1a_soup.get_text("\n\n")
    remaining_text = remaining_soup.get_text("\n\n")

    return item_1a_text, remaining_text, True
    

# Function to calculate performance
def calculate_performance(company, period):
    """Calculate the performance percentage."""
    try:
        base_price = float(company['stock_prices']["At Filing Date"])
        current_price = float(company['stock_prices'][period])
        return (current_price - base_price) / base_price * 100
    except (ValueError, TypeError):
        return None

# Update stock data
for industry, companies in industry_companies.items():
    dl = initialize_downloader(industry)
    #for company in companies[:5]:  # Limiting to first 5 companies for testing
    for company in companies: 
        download_and_extract_item1a(company, dl)
        if not company['filing_date']:
            continue

        try:
            filing_date_obj = datetime.strptime(company['filing_date'], "%Y-%m-%d")
            print(f"Filing Date Object: {filing_date_obj}")
        except ValueError:
            continue

        if not company['yahoo_ticker']:
            continue

        dates = {
            "At Filing Date": filing_date_obj,
            "1 Year After": filing_date_obj + timedelta(days=365),
        }

        stock = yf.Ticker(company['yahoo_ticker'])
        prices = {}

        for label, target_date in dates.items():
            start_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")
            print(f"{label} - Start Date: {start_date}, End Date: {end_date}")
            data = stock.history(start=start_date, end=end_date)

            if not data.empty:
                data.index = data.index.tz_localize(None)
                closest_date = min(data.index, key=lambda x: abs(x - target_date))
                price = data.loc[closest_date, 'Close']
            else:
                price = "N/A"
            prices[label] = price

        company['stock_prices'] = prices

# Filter companies with complete data
complete_companies = [company for industry, companies in industry_companies.items() for company in companies if all(company['stock_prices'][period] not in [None, "N/A"] for period in ["At Filing Date", "1 Year After"])]

# Calculate performances and sentiments
time_periods = ["1 Year After"]
individual_performances = {period: {} for period in time_periods}
portfolio_performances = {period: [] for period in time_periods}

for company in complete_companies:
    for period in time_periods:
        perf = calculate_performance(company, period)
        if perf is not None:
            individual_performances[period][company['name']] = perf
            portfolio_performances[period].append(perf)
            company[f'{period} Performance (%)'] = perf
            company['Sentiment'] = "Positive" if perf > 0 else "Negative"

# Calculate overall portfolio performance
portfolio_avg_performance = {period: np.mean(perfs) if perfs else 0 for period, perfs in portfolio_performances.items()}

# Graphical Overview
for period in time_periods:
    performances = individual_performances[period]
    names = list(performances.keys())
    values = list(performances.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)

    # Highlight the average portfolio performance
    plt.axhline(y=portfolio_avg_performance[period], color='r', linestyle='--', label='Portfolio Average')

    plt.xlabel('Company')
    plt.ylabel('Performance (%)')
    plt.title(f'Stock Performance: {period}')
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Color bars based on comparison with the portfolio average
    for bar, value in zip(bars, values):
        bar.set_color('green' if value > portfolio_avg_performance[period] else 'red')

    plt.show()

# Save results to Excel file
with pd.ExcelWriter('Data/companies_stock_data.xlsx', engine='openpyxl') as writer:
    for industry, companies in industry_companies.items():
        df = pd.DataFrame(companies)
        df.to_excel(writer, sheet_name=industry, index=False)

print("Data has been saved to companies_stock_data.xlsx")


