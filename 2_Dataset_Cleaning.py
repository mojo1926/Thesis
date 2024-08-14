import pandas as pd

# Load the existing Excel file
excel_file_path = 'C:/Users/antho/OneDrive/Masterthesis/Thesis/Data/companies_stock_data.xlsx'
xls = pd.ExcelFile(excel_file_path)

# Read the 'service' sheet into a DataFrame
service_df = pd.read_excel(xls, sheet_name='service')

# Remove the specified columns
service_df = service_df.drop(columns=['yahoo_ticker', 'has_item1a', 'industry','stock_prices'])

# Remove duplicate companies based on 'name' column
service_df = service_df.drop_duplicates(subset='name')

# Remove companies with incomplete data (N/A or None values)
service_df = service_df.dropna(subset=['1 Year After Performance (%)'])

# Remove companies without Item 1A
service_df = service_df.dropna(subset=['sic', 'item1a_content'])

# Save the cleaned DataFrame to a new Excel file
output_file_path = 'C:/Users/antho/OneDrive/Masterthesis/Thesis/Data/cleaned_service_companies_stock_data.xlsx'

with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    service_df.to_excel(writer, sheet_name='service', index=False)

print(f"Cleaned data has been saved to {output_file_path}")