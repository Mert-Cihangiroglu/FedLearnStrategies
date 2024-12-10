# visualization.py
import matplotlib.pyplot as plt
import json

def plot_accuracy_trends(results_file, output_file):
    """
    Plot accuracy trends across federated learning experiments.

    Parameters:
        results_file (str): Path to the JSON file containing experiment results.
        output_file (str): Path to save the output plot.
    """
    with open(results_file, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(12, 8))

    for experiment in results:
        config = experiment["experiment"]
        metrics = experiment["metrics"]

        aggregation_method = config["aggregation"]
        alpha = config.get("alpha", "IID")

        accuracy = [round_metric["accuracy"] for round_metric in metrics]
        plt.plot(accuracy, label=f"{aggregation_method} (alpha={alpha})")

    plt.title("Accuracy Trends Across Federated Learning Experiments")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    plot_accuracy_trends(results_file="results.json", output_file="accuracy_trends.png")
