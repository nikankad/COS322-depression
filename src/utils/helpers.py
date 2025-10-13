from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#summarize model metric in string format and saves sample_submission_{modelname_roc_mse_r^2}.csv. id will just be the currentid+1
def print_output(model_name, y_pred, y_test):
    """Print model performance metrics in a clean format."""
    print(f"\n📊 Results for {model_name}:")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    # Root Mean Squared Error
    rmse = mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # R² Score (coefficient of determination)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2}")




    #generate sample_submission_{modelname_roc_mse_r^2}.csv


def pred_actual_graph():
    #will output a graph of pred vs actual to see how our model did 
    print("d")