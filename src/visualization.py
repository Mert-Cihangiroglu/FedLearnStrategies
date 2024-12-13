import os
import json
import matplotlib.pyplot as plt

def plot_accuracy_trends(results_dir, output_file=None):
    """
    Plot accuracy trends across federated learning experiments.

    Parameters:
        results_dir (str): Path to the directory containing JSON result files.
        output_file (str, optional): Path to save the combined output plot.
    """
    plt.figure(figsize=(12, 8))

    for result_file in sorted(os.listdir(results_dir)):
        if result_file.endswith(".json"):
            file_path = os.path.join(results_dir, result_file)
            with open(file_path, "r") as f:
                results = json.load(f)

            config = results["experiment"]
            metrics = results["metrics"]

            aggregation_method = config["aggregation"]
            dataset = config["dataset"]
            alpha = config.get("alpha", "IID")

            accuracy = [round_metric["accuracy"] for round_metric in metrics]
            plt.plot(accuracy, label=f"{dataset} | {aggregation_method} (alpha={alpha})")

    plt.title("Accuracy Trends Across Federated Learning Experiments")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Combined plot saved to {output_file}")
    else:
        plt.show()

def plot_individual_accuracy_trends(results_dir, output_dir):
    """
    Plot individual accuracy trends for each experiment.

    Parameters:
        results_dir (str): Path to the directory containing JSON result files.
        output_dir (str): Path to the directory to save individual plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for result_file in sorted(os.listdir(results_dir)):
        if result_file.endswith(".json"):
            file_path = os.path.join(results_dir, result_file)
            with open(file_path, "r") as f:
                results = json.load(f)

            config = results["experiment"]
            metrics = results["metrics"]

            aggregation_method = config["aggregation"]
            dataset = config["dataset"]
            alpha = config.get("alpha", "IID")

            accuracy = [round_metric["accuracy"] for round_metric in metrics]

            plt.figure(figsize=(10, 6))
            plt.plot(accuracy, label=f"{dataset} | {aggregation_method} (alpha={alpha})")
            plt.title(f"Accuracy Trend: {dataset} | {aggregation_method} (alpha={alpha})")
            plt.xlabel("Round")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_filename = f"{dataset}_alpha{alpha}_{aggregation_method}.png".replace(".", "_")
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Individual plot saved to {plot_path}")

# Example Usage
if __name__ == "__main__":
    # Plot combined trends
    plot_accuracy_trends(results_dir="results", output_file="combined_accuracy_trends.png")
    
    # Plot individual trends
    plot_individual_accuracy_trends(results_dir="results", output_dir="individual_plots")