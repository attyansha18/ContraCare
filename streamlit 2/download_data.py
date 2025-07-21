import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def download_and_prepare_data():
    # Check if data file exists
    if not os.path.exists('cmc.data'):
        print("Downloading CMC dataset...")
        # Download the dataset from UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"
        df = pd.read_csv(url, header=None)
        df.to_csv('cmc.data', index=False, header=False)
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists. Loading from file...")
        df = pd.read_csv('cmc.data', header=None)
    
    # Set column names based on dataset description
    df.columns = [
        'wife_age', 'wife_education', 'husband_education', 'num_children',
        'wife_religion', 'wife_now_working', 'husband_occupation',
        'standard_of_living', 'media_exposure', 'contraceptive_method'
    ]
    
    # Save the processed data
    df.to_csv('processed_cmc.csv', index=False)
    print("Data processed and saved successfully!")
    
    return df

if __name__ == "__main__":
    download_and_prepare_data() 