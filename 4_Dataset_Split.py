#Training 72%
#Validation 18%
#Test 10%

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned Excel file
input_file_path = 'Data/All_Service_Risks.xlsx'
service_df = pd.read_excel(input_file_path, sheet_name='service')

# Set the random seed for reproducibility
random_seed = 42

# Split the data into training (72%), validation (18%), and test (10%) sets
train_df, temp_df = train_test_split(service_df, test_size=0.28, random_state=random_seed)
val_df, test_df = train_test_split(temp_df, test_size=(10/28), random_state=random_seed)

# Save the datasets to separate Excel files
train_file_path = 'Data/train_service_risks.xlsx'
val_file_path = 'Data/val_service_risks.xlsx'
test_file_path = 'Data/test_service_risks.xlsx'

with pd.ExcelWriter(train_file_path, engine='openpyxl') as writer:
    train_df.to_excel(writer, sheet_name='service', index=False)

with pd.ExcelWriter(val_file_path, engine='openpyxl') as writer:
    val_df.to_excel(writer, sheet_name='service', index=False)

with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
    test_df.to_excel(writer, sheet_name='service', index=False)

print(f"Training data has been saved to {train_file_path}")
print(f"Validation data has been saved to {val_file_path}")
print(f"Test data has been saved to {test_file_path}")