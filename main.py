# main.py
import torch
from device import get_device
from data_preparation import load_dataset, partition_data_iid, partition_data_dirichlet, create_data_loaders
from model import get_model
from client import Client
from validator import Validator
from aggregation import fedavg_aggregation, fedprox_aggregation, fednova_aggregation, scaffold_aggregation
import warnings
warnings.filterwarnings("ignore")


# Set up device
device = get_device()

# Mapping datasets to the number of output classes
DATASET_OUTPUT_CLASSES = {
    "MNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "TinyImageNet": 200
}

DATASET_INPUT_CHANNELS = {
    "MNIST": 1,        # Grayscale images
    "CIFAR10": 3,      # RGB images
    "CIFAR100": 3,     # RGB images
    "TinyImageNet": 3  # RGB images
}

# Experiment configurations
experiments = [
    {
        "dataset": dataset,
        "batch_size": 64,
        "data_distribution": "Dirichlet",
        "alpha": alpha,
        "num_clients": 10,
        "num_rounds": 100,
        "local_epochs": 10,
        "learning_rate": 0.01,
        "aggregation": method,
        "output_classes": DATASET_OUTPUT_CLASSES[dataset],  # Dynamically set output classes
        "mu": 0.01 if method == "FedProx" else 0.0
    }
    for dataset in [ "CIFAR100", "TinyImageNet"]  # Add datasets dynamically
    for alpha in [0.125, 0.3, 0.5, 0.75, 1.0]
    for method in ["FedAvg", "FedProx", "FedNova", "SCAFFOLD"]
]

results = []

# Run experiments
for exp_idx, config in enumerate(experiments):
    print(f"Starting experiment {exp_idx + 1}/{len(experiments)}: Alpha={config['alpha']}, Method={config['aggregation']}...")

    # Load dataset
    dataset, test_loader = load_dataset(config["dataset"], config["batch_size"])

    # Partition dataset
    if config["data_distribution"] == "IID":
        client_data = partition_data_iid(dataset, config["num_clients"])
    elif config["data_distribution"] == "Dirichlet":
        client_data = partition_data_dirichlet(dataset, config["num_clients"], config["alpha"])
    else:
        raise ValueError("Unsupported data distribution")

    # Create client data loaders
    client_loaders = create_data_loaders(client_data, config["batch_size"])

    # Initialize clients
    clients = [
        Client(client_id=i, data_loader=client_loaders[i], device=device)
        for i in range(config["num_clients"])
    ]

    # Dynamically create the model
    dataset_name = config["dataset"]
    input_channels = DATASET_INPUT_CHANNELS[dataset_name]
    output_classes = config["output_classes"]

    global_model = get_model(output_classes=output_classes, input_channels=input_channels)
    validator = Validator(test_loader=test_loader, device=device)

    # Run federated learning rounds
    aggregation_method = config["aggregation"]
    global_control_variates = None

    round_metrics = []

    for round_num in range(config["num_rounds"]):
        print(f"Round {round_num + 1}...")

        # Local training
        client_models = []
        client_sizes = []
        control_variates = []
        for client in clients:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=config["learning_rate"])
            criterion = torch.nn.CrossEntropyLoss()
            strategy_params = {"mu": config.get("mu", 0.0)} if aggregation_method == "FedProx" else {}

            local_model = client.local_train(global_model, config["local_epochs"], criterion, optimizer, strategy_params)
            client_models.append(local_model)
            client_sizes.append(len(client.data_loader.dataset))

            # the control variates are initialized and passed between clients and the server.
            if aggregation_method == "SCAFFOLD":
                control_variates.append(client.control_variates)

        # Aggregation
        if aggregation_method == "FedAvg":
            global_model.load_state_dict(fedavg_aggregation(client_models, client_sizes))
        elif aggregation_method == "FedProx":
            global_model.load_state_dict(fedprox_aggregation(client_models, client_sizes, config["mu"]))
        elif aggregation_method == "FedNova":
            global_model.load_state_dict(fednova_aggregation(client_models, client_sizes, [1.0] * len(clients)))
        elif aggregation_method == "SCAFFOLD":
            global_model_state, global_control_variates = scaffold_aggregation(client_models, client_sizes, control_variates)
            global_model.load_state_dict(global_model_state)

        # Validation
        metrics = validator.validate(global_model)
        validator.log_results(metrics, round_num + 1)
        round_metrics.append(metrics)

    # Record results
    results.append({
        "experiment": config,
        "metrics": round_metrics
    })

    print(f"Experiment {exp_idx + 1} complete.\n")

# Save all results
with open("results.json", "w") as results_file:
    import json
    json.dump(results, results_file)

print("All experiments complete. Results saved to results.json.")