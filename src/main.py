# main.py
import torch
from device import get_device
from data_preparation import load_dataset, partition_data_iid, partition_data_dirichlet, create_data_loaders
from model import get_model
from client import Client
from validator import Validator
from aggregation import fedavg_aggregation, fedprox_aggregation, fednova_aggregation, scaffold_aggregation
from backdoor_strategy import BackdoorStrategy
import warnings
import random
import os
warnings.filterwarnings("ignore")

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


# Set up device
device = get_device()

# Mapping datasets to the number of output classes
DATASET_OUTPUT_CLASSES = {
    "MNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "TinyImageNet": 200,
    "pathmnist": 9,                # 9 tissue types
    "organmnist_axial": 11,        # 11 organ types
    "organmnist_coronal": 11,      # 11 organ types
    "organmnist_sagittal": 11      # 11 organ types
}

DATASET_INPUT_CHANNELS = {
    "MNIST": 1,                   # Grayscale images
    "CIFAR10": 3,                 # RGB images
    "CIFAR100": 3,                # RGB images
    "TinyImageNet": 3,            # RGB images
    "pathmnist": 3,               # RGB images (3 channels)
    "organmnist_axial": 1,        # Grayscale (1 channel)
    "organmnist_coronal": 1,      # Grayscale (1 channel)
    "organmnist_sagittal": 1      # Grayscale (1 channel)
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
        "mu": 0.01 if method == "FedProx" else 0.0,
        "attack_type": "None",  # Specify attack type (None for no attack, "backdoor" for backdoor attacks)
        "malicious_client_percentage": 0.2,  # 20% malicious clients
        "backdoor_target": 7,  # Target label for backdoor attack
        "backdoor_percentage": 0.3  # 30% of data poisoned for malicious clients
    }
    for dataset in ["pathmnist","organmnist_axial", "organmnist_coronal", "organmnist_sagittal","CIFAR10", "CIFAR100", "TinyImageNet"]
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
    
    # Initialize attack settings
    malicious_client_ids = []
    backdoor_strategy = None
    
    if config["attack_type"] == "backdoor":
        # Initialize backdoor strategy
        backdoor_strategy = BackdoorStrategy(trigger_type="square", triggerX=25, triggerY=25)

        # Randomly select malicious clients
        num_malicious_clients = int(config["num_clients"] * config["malicious_client_percentage"])
        malicious_client_ids = random.sample(range(config["num_clients"]), num_malicious_clients)

    # Initialize clients
    clients = []
    for i in range(config["num_clients"]):
        is_malicious = i in malicious_client_ids
        clients.append(Client(
            client_id=i,
            data_loader=client_loaders[i],
            device=device,
            is_malicious=is_malicious,
            attack_type=config["attack_type"] if is_malicious else None,
            backdoor_strategy=backdoor_strategy if is_malicious else None,
            backdoor_percentage=config["backdoor_percentage"] if is_malicious else 0.0,
            backdoor_target=config["backdoor_target"] if is_malicious else None
        ))
    
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
    
        # Save results for this experiment
    experiment_result = {
        "experiment": config,
        "metrics": round_metrics
    }
    
    # Create unique filename based on experiment configuration
    filename = f"results/results_{config['dataset']}_alpha{config['alpha']}_method{config['aggregation']}_isAttack_{config['attack_type']}.json".replace(".", "_")
    with open(filename, "w") as results_file:
        import json
        json.dump(experiment_result, results_file)

    print(f"Experiment {exp_idx + 1} complete. Results saved to {filename}\n")


print("All experiments complete. Results saved to results.json.")