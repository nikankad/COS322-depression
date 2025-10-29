from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import os
from dotenv import load_dotenv

#students_dfsummarize model metric in string format and saves sample_submission_{modelname_roc_mse_r^2}.csv
def regression_metrics(model_name, y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    """Print model performance metrics in a clean format."""
    print(f"\n📊 Results for {model_name}:")

    print(f"Mean Absolute Error (MAE): {mae}")
    # Mean Squared Error
    print(f"Mean Squared Error (MSE): {mse}")

    # Root Mean Squared Error
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # R² Score (coefficient of determination)
    print(f"R² Score: {r2}")

def classification_metric(model_name, y_pred, y_test):
    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score 
    )  
        # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


    # Print metrics
    print("\n📊 SVC Model Performance")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 40)
    

def regression_graph(y_pred, y_test):
    plt.figure(figsize=(7, 5))
    # Scatter actual vs predicted
    plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6, label="Predicted vs Actual")
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Fit")

    # plt.title(f"{type} — Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def generate_classification_charts(y_pred, y_test, y_pred_prob):
        cm = sk_confusion_matrix(y_test, y_pred)

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()

def preprocessing(df: pd.DataFrame)->pd.DataFrame:
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        """
        Preprocess the input DataFrame for model training.

        Steps:
        - Convert 'Yes'/'No' to 1/0
        - Encode 'working_student', 'gender', 'dietary_habits', 'sleep_duration', 'degree', 'profession'
        - Remove rare categories (less than 10 occurrences)
        - Fill NaN professions
        """

        #rename columns
        df.rename(columns={'Working Professional or Student': 'Working Student', 'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts', 'Family History of Mental Illness':'Family Mental Illness' }, inplace=True)

        # Convert all column names to snake_case
        df.columns = (
            df.columns
            .str.strip()                              # remove leading/trailing spaces
            .str.replace(' ', '_')                    # replace spaces with underscores
            .str.replace('[^A-Za-z0-9_]+', '', regex=True)  # remove special characters
            .str.lower()            
                            # convert to lowercase (optional)
        )

        # Convert Yes/No columns to binary
        df = df.map(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else x)

        # Encode working_student: 1 = Working Professional, 0 = Student
        if 'working_student' in df.columns:
            df['working_student'] = df['working_student'].map({'Working Professional': 1, 'Student': 0})

        # Encode gender: Male = 1, Female = 0
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # Remove rare sleep_duration values (<10 occurrences)
        if 'sleep_duration' in df.columns:
            counts = df['sleep_duration'].value_counts()
            df = df[df['sleep_duration'].isin(counts[counts >= 10].index)]

        # Remove rare dietary_habits values (<10 occurrences)
        if 'dietary_habits' in df.columns:
            counts = df['dietary_habits'].value_counts()
            df = df[df['dietary_habits'].isin(counts[counts >= 10].index)]

        # Map dietary_habits to numeric
        df['dietary_habits'] = df['dietary_habits'].map({
            "Unhealthy": 0,
            "Moderate": 1,
            "Healthy": 2
        })

        # If working_student == 0, set profession to "Student"
        if 'working_student' in df.columns and 'profession' in df.columns:
            df.loc[df["working_student"] == 0, "profession"] = "Student"

        # Fill missing professions
        if 'profession' in df.columns:
            df.loc[df["profession"].isna(), "profession"] = "Unemployed"

        # Encode categorical columns
        le = LabelEncoder()
        if 'sleep_duration' in df.columns:
            df['sleep_duration'] = le.fit_transform(df['sleep_duration'].astype(str))
        if 'degree' in df.columns:
            df['degree'] = le.fit_transform(df['degree'].astype(str))
        if 'profession' in df.columns:
            df['profession'] = le.fit_transform(df['profession'].astype(str))

        # #scale our data
        # scaler = StandardScaler()
        # numeric_cols = df.select_dtypes(include=["number"]).columns.drop({'id', 'gender', 'suicidal_thoughts', 'family_mental_illness', 'depression', 'working_student'})
        # df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) 

        # # Clean column names
        # load_dotenv()
        # gdpdf = pd.read_csv(os.environ['GDP_LOCATION'])

        # gdpdf.columns = gdpdf.columns.str.strip()
        # gdpdf.rename(columns=lambda x: x.replace('\xa0', ' '), inplace=True)

        # # Group GDP data by city name and sum GDP
        # gdp_summed = (
        #     gdpdf.groupby("Metropolitan area", as_index=False)["Nominal GDP"]
        #     .sum()
        #     .rename(columns={"Metropolitan area": "city", "Nominal GDP": "total_gdp"})
        # )


        # # Merge into your existing cities_df
        # df = df.merge(gdp_summed, on="city", how="left")

        return df