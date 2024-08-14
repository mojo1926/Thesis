import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define file paths
labeled_risks_input_path = 'Risks/labeled_risks.xlsx'
evaluation_output_path = 'Risks/Sentiment_Comparison.xlsx'

def main_comparison(labeled_risks_input_path, evaluation_output_path):
    # Load data from the labeled risks Excel file
    df = pd.read_excel(labeled_risks_input_path, sheet_name=None)
    
    all_comparisons = []

    tickers = df['Filed Risks']['Ticker'].unique()

    for ticker in tickers:
        print(f"Starting ticker {ticker}")
        
        # Retrieve sentiments for each category
        artificial_sentiments = df['Identified Risks'][df['Identified Risks']['Ticker'] == ticker]['Sentiment'].tolist()
        baseline_sentiments = df['Random Risks'][df['Random Risks']['Ticker'] == ticker]['Sentiment'].tolist()
        top_line_sentiments = df['Filed Risks'][df['Filed Risks']['Ticker'] == ticker]['Sentiment'].tolist()

        # Check for empty sentiment lists and skip tickers with missing sentiments
        if not artificial_sentiments or not any(artificial_sentiments):
            print(f"Skipping ticker {ticker} due to missing or zero artificial sentiments.")
            continue
        if not top_line_sentiments or not any(top_line_sentiments):
            print(f"Skipping ticker {ticker} due to missing or zero top-line sentiments.")
            continue
        if not baseline_sentiments or not any(baseline_sentiments):
            print(f"Skipping ticker {ticker} due to missing or zero baseline sentiments.")
            continue

        # Aggregate sentiments: calculate the mean sentiment for artificial, top-line, and baseline risks
        artificial_avg_sentiment = np.mean(artificial_sentiments) if artificial_sentiments else 0
        top_line_avg_sentiment = np.mean(top_line_sentiments) if top_line_sentiments else 0
        baseline_avg_sentiment = np.mean(baseline_sentiments) if baseline_sentiments else 0

        comparison = {
            'Ticker': ticker,
            'Artificial Sentiment': artificial_avg_sentiment,
            'Top-Line Sentiment': top_line_avg_sentiment,
            'Baseline Sentiment': baseline_avg_sentiment
        }
        all_comparisons.append(comparison)

    comparison_df = pd.DataFrame(all_comparisons)

    with pd.ExcelWriter(evaluation_output_path) as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison Results', index=False)

    return comparison_df

def load_data_and_perform_ttest(file_path):
    # Load data
    df = pd.read_excel(file_path, sheet_name='Comparison Results')

    # Calculate errors
    df['Error_Artificial'] = np.abs(df['Top-Line Sentiment'] - df['Artificial Sentiment'])
    df['Error_Baseline'] = np.abs(df['Top-Line Sentiment'] - df['Baseline Sentiment'])

    # T-tests
    # T-test 1: Error comparisons between Artificial and Baseline
    t_stat1, p_value1 = stats.ttest_ind(df['Error_Artificial'], df['Error_Baseline'])
    print("T-test 1 Results (Artificial vs. Baseline):")
    print(f"T-statistic: {t_stat1}, P-value: {p_value1}")

    # T-test 2: Artificial vs. Top-Line
    t_stat2, p_value2 = stats.ttest_ind(df['Artificial Sentiment'], df['Top-Line Sentiment'])
    print("T-test 2 Results (Artificial vs. Top-Line):")
    print(f"T-statistic: {t_stat2}, P-value: {p_value2}")

    # RMSE
    rmse_artificial = np.sqrt(mean_squared_error(df['Top-Line Sentiment'], df['Artificial Sentiment']))
    rmse_baseline = np.sqrt(mean_squared_error(df['Top-Line Sentiment'], df['Baseline Sentiment']))

    # MAE
    mae_artificial = mean_absolute_error(df['Top-Line Sentiment'], df['Artificial Sentiment'])
    mae_baseline = mean_absolute_error(df['Top-Line Sentiment'], df['Baseline Sentiment'])

    print("RMSE (Artificial):", rmse_artificial)
    print("RMSE (Baseline):", rmse_baseline)
    print("MAE (Artificial):", mae_artificial)
    print("MAE (Baseline):", mae_baseline)

# Run the comparison and evaluation
comparison_df = main_comparison(labeled_risks_input_path, evaluation_output_path)
load_data_and_perform_ttest(evaluation_output_path)
