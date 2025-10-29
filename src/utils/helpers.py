
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
def preprocessing(df):
    # Preprocessing
    #rename columns
    df.rename(columns={'Working Professional or Student': 'Working Student', 'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts', 'Family History of Mental Illness':'Family Mental Illness' }, inplace=True)

    # Convert all column names to snake_case
    df.columns = (
        df.columns
        .str.strip()                              # remove leading/trailing spaces
        .str.replace(' ', '_')                    # replace spaces with underscores
        .str.replace('[^A-Za-z0-9_]+', '', regex=True)  # remove special characters
        .str.lower()            # convert to lowercase
                         
    )
    # Convert Yes/No columns to binary (1/0)
    df = df.map(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else x)
    # Make working_student binary: 1 if working, 0 if student
    df['working_student'] = df['working_student'].map({'Working Professional': 1, 'Student': 0})
    # Convert gender to binary: Male = 1, Female = 0
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    # remove sleep_duration occurences that have less than 10 appearences
    # Replace sleep_duration occurrences that have less than 10 appearances with the most common value
    sleep_duration_mode = df['sleep_duration'].mode()[0]
    df['sleep_duration'] = df['sleep_duration'].where(df['sleep_duration'].isin(df['sleep_duration'].value_counts()[df['sleep_duration'].value_counts() >= 10].index), sleep_duration_mode)

    # Replace dietary_habits occurrences that have less than 10 appearances with the most common value
    dietary_habits_mode = df['dietary_habits'].mode()[0]
    df['dietary_habits'] = df['dietary_habits'].where(df['dietary_habits'].isin(df['dietary_habits'].value_counts()[df['dietary_habits'].value_counts() >= 10].index), dietary_habits_mode)
    df["dietary_habits"] = df["dietary_habits"].map({
        "Unhealthy": 0,
        "Moderate": 1,
        "Healthy": 2
    })
    #if profession is student then make profession "Student"
    df.loc[df["working_student"] == 0, "profession"] = "Student"
    #If profession is still NaN, set to "Unemployed"
    df.loc[df["profession"].isna(), "profession"] = "Unemployed"
    #Map sleeping hours to numbers
    df['sleep_duration'] = LabelEncoder().fit_transform(df['sleep_duration'])
    df['degree'] = LabelEncoder().fit_transform(df['degree'])
    # df['profession'] = LabelEncoder().fit_transform(df['degree'])
    df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
    # df.dropna(inplace=True)
    # df.fillna(0, inplace=True)
    
    scaler = StandardScaler()
    # Scale only numeric columns that are not binary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_binary_cols = [c for c in numeric_cols if set(df[c].dropna().unique()) != {0, 1}]

    df[non_binary_cols] = scaler.fit_transform(df[non_binary_cols])

    return df

import os
import pandas as pd
def generate_submission(y_pred):
    """
    Generates a CSV file with columns 'ids' and 'depression'
    saved to the directory in the environment variable RESULSTS_LOCATION.
    Expects y_pred to be a DataFrame or array-like with 'id' and 'y_pred' columns.
    """
    results_dir = os.getenv("RESULTS_LOCATION")
    if not results_dir:
        raise EnvironmentError("RESULTS_LOCATION not set in environment variables.")
    
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "submission.csv")

    # Handle both DataFrame and tuple inputs
    if isinstance(y_pred, pd.DataFrame):
        df = y_pred.rename(columns={"id": "id", "y_pred": "depression"})
    else:
        raise TypeError("y_pred must be a DataFrame with 'id' and 'y_pred' columns.")
    
    df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    

