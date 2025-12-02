# from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt 
import seaborn as sns

def prepare_xy(df: pd.DataFrame):
        # """ Prepare X, y from df: drop NA, select numeric cols, handle id if present."""

        numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

        X = numeric_df.drop(columns=["depression", "id"])
        y = numeric_df["depression"]

        return X, y


def preprocessing(df):
    # Preprocessing
    # rename columns
    df.rename(
        columns={
            "Working Professional or Student": "Working Student",
            "Have you ever had suicidal thoughts ?": "Suicidal Thoughts",
            "Family History of Mental Illness": "Family Mental Illness",
        },
        inplace=True,
    )

    # Convert all column names to snake_case
    df.columns = (
        df.columns.str.strip()  # remove leading/trailing spaces
        .str.replace(" ", "_")  # replace spaces with underscores
        .str.replace("[^A-Za-z0-9_]+", "", regex=True)  # remove special characters
        .str.lower()  # convert to lowercase
    )
    # Convert Yes/No columns to binary (1/0)
    df = df.map(lambda x: 1 if x == "Yes" else 0 if x == "No" else x)
    # Make working_student binary: 1 if working, 0 if student
    df["working_student"] = df["working_student"].map(
        {"Working Professional": 1, "Student": 0}
    )
    # Convert gender to binary: Male = 1, Female = 0
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    # remove sleep_duration occurences that have less than 10 appearences
    # Replace sleep_duration occurrences that have less than 10 appearances with the most common value
    sleep_duration_mode = df["sleep_duration"].mode()[0]
    df["sleep_duration"] = df["sleep_duration"].where(
        df["sleep_duration"].isin(
            df["sleep_duration"]
            .value_counts()[df["sleep_duration"].value_counts() >= 10]
            .index
        ),
        sleep_duration_mode,
    )

    # Replace dietary_habits occurrences that have less than 10 appearances with the most common value
    dietary_habits_mode = df["dietary_habits"].mode()[0]
    df["dietary_habits"] = df["dietary_habits"].where(
        df["dietary_habits"].isin(
            df["dietary_habits"]
            .value_counts()[df["dietary_habits"].value_counts() >= 10]
            .index
        ),
        dietary_habits_mode,
    )
    df["dietary_habits"] = df["dietary_habits"].map(
        {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
    )

    # if profession is student then make profession "Student"
    df.loc[df["working_student"] == 0, "profession"] = "Student"
    # If profession is still NaN, set to "Unemployed"
    df.loc[df["profession"].isna(), "profession"] = "Unemployed"

    # gdp_df = pd.read_csv(
    #     "/Users/nikan/Desktop/School/Sems/Spring 2025/COS 322/COS322-depression/data/ExtraData/CityGDP.csv"
    # )

    # # Normalize city names for reliable merge
    # df["city"] = df["city"].str.strip().str.lower()
    # gdp_df["city"] = gdp_df["city"].str.strip().str.lower()

    # # Merge GDP info
    # df = df.merge(gdp_df[["city", "gdp", "ppp"]], on="city", how="left")

    sleep = {
        "More than 8 hours": 9,
        "Less than 5 hours": 4,
        "5-6 hours": 5.5,
        "7-8 hours": 7.5,
        "1-2 hours": 1.5,
        "6-8 hours": 7,
        "4-6 hours": 5,
        "6-7 hours": 6.5,
        "10-11 hours": 10.5,
        "8-9 hours": 8.5,
        "9-11 hours": 10,
        "2-3 hours": 2.5,
        "3-4 hours": 3.5,
        "Moderate": 6,
        "4-5 hours": 4.5,
        "9-6 hours": 7.5,
        "1-3 hours": 2,
        "1-6 hours": 4,
        "8 hours": 8,
        "10-6 hours": 8,
        "Unhealthy": 3,
        "Work_Study_Hours": 6,
        "3-6 hours": 3.5,
        "9-5": 7,
        "9-5 hours": 7,
    }
    df["sleep_duration"] = df["sleep_duration"].map(sleep)
    df.loc[:, "sleep_duration"] = df["sleep_duration"].fillna(
        df["sleep_duration"].mode()[0]
    )

    # Apply Label Encoding to sleep_duration
    label_encoder = LabelEncoder()
    # Degree
    df["degree"] = df["degree"].astype(str).fillna("Unknown")
    df["degree"] = label_encoder.fit_transform(df["degree"])

    # df["city"] = df["city"].astype(str).fillna("Unknown")
    # df["city"] = label_encoder.fit_transform(df["city"])
    # Profession
    df["profession"] = df["profession"].astype(str).fillna("Unknown")
    df["profession"] = label_encoder.fit_transform(df["profession"])

    df.fillna(df.select_dtypes(include=["number"]).median(), inplace=True)

    # # combine pressure columns
    # df["pressure"] = df["academic_pressure"] + df["work_pressure"]
    df.drop(columns=["name", "city"], inplace=True)
    # # Combine satisfaction columns
    # df["satisfaction"] = df["study_satisfaction"] + df["job_satisfaction"]
    # df.drop(columns=["study_satisfaction", "study_satisfaction"], inplace=True)

    # scale

    # Identify numeric columns safely
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [
        c
        for c in numeric_cols
        if c
        not in [
            "id",
            "depression",
            # "gender",
            # "family_mental_illness",
            # "working_student",
            # "suicidal_thoughts",
        ]
    ]

    # Convert possible string numerics
    df[cols_to_scale] = df[cols_to_scale].apply(pd.to_numeric, errors="coerce")

    # Fill missing after coercion
    df[cols_to_scale] = df[cols_to_scale].fillna(df[cols_to_scale].median())

    # Scale
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df


def generate_submission(df, modelName):
    """
    Generates a CSV file with columns 'ids' and 'depression'
    saved to the directory in the environment variable RESULSTS_LOCATION.
    Expects y_pred to be a DataFrame or array-like with 'id' and 'y_pred' columns.
    """
    results_dir = os.getenv("SUBMISSION_LOCATION")
    if not results_dir:
        raise EnvironmentError("SUBMISSION_LOCATION not set in environment variables.")

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(
        results_dir, f"submission_{modelName}_{df['id'].iloc[0]}.csv"
    )

    # Handle both DataFrame and tuple inputs
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns={"id": "id", "y_pred": "depression"})
    else:
        raise TypeError("y_pred must be a DataFrame with 'id' and 'y_pred' columns.")

    df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")

def report_metrics(model, threshold, X_test, y_test):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)

        print(classification_report(y_test, y_pred))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        ax[0].set_title("ROC Curve")
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].legend()

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1])
        ax[1].set_title(f"Confusion Matrix (threshold={threshold:.3f})")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")

        plt.tight_layout()
        plt.show()


def find_best_threshold(model, X_test, y_test):
        probs = model.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)

        best_idx = np.argmax(f1_scores)

        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        return best_threshold, best_f1, best_precision, best_recall
