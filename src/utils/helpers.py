#summarize model metric in string format and saves sample_submission_{r^2 score}.csv. id will just be the currentid+1
def output(model_name, metrics: dict):
    """Print model performance metrics in a clean format."""
    print(f"\n📊 Results for {model_name}:")
    for metric, value in metrics.items():
        print(f" - {metric.capitalize()}: {value:.3f}")

    #generate sample_submission_{r^2 score}.csv


def pred_actual_graph():
    #will output a graph of pred vs actual to see how our model did 