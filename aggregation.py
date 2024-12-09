# aggregation.py
import torch

def fedavg_aggregation(client_models, client_sizes):
    """
    Perform FedAvg aggregation by averaging client models.

    Parameters:
        client_models (list[dict]): List of state_dicts of client models.
        client_sizes (list[int]): List of data sizes corresponding to each client.

    Returns:
        dict: Aggregated global model state.
    """
    global_model = {}
    total_samples = sum(client_sizes)

    for key in client_models[0].keys():
        global_model[key] = sum(client_sizes[i] * client_models[i][key] for i in range(len(client_models))) / total_samples

    return global_model

def fedprox_aggregation(client_models, client_sizes, mu):
    """
    Perform FedProx aggregation, incorporating proximal term.

    Parameters:
        client_models (list[dict]): List of state_dicts of client models.
        client_sizes (list[int]): List of data sizes corresponding to each client.
        mu (float): Proximal term coefficient.

    Returns:
        dict: Aggregated global model state.
    """
    return fedavg_aggregation(client_models, client_sizes)  # FedProx uses the same aggregation as FedAvg

def fednova_aggregation(client_models, client_sizes, update_norms):
    """
    Perform FedNova aggregation, normalizing weights based on client contributions.

    Parameters:
        client_models (list[dict]): List of state_dicts of client models.
        client_sizes (list[int]): List of data sizes corresponding to each client.
        update_norms (list[float]): Normalized update contributions for each client.

    Returns:
        dict: Aggregated global model state.
    """
    global_model = {}
    total_norm = sum(update_norms)

    for key in client_models[0].keys():
        global_model[key] = sum(update_norms[i] * client_models[i][key] for i in range(len(client_models))) / total_norm

    return global_model

def scaffold_aggregation(client_models, client_sizes, control_variates):
    """
    Perform SCAFFOLD aggregation using control variates for variance reduction.

    Parameters:
        client_models (list[dict]): List of state_dicts of client models.
        client_sizes (list[int]): List of data sizes corresponding to each client.
        control_variates (list[dict]): Control variates from clients.

    Returns:
        dict: Aggregated global model state with updated control variates.
    """
    global_model = fedavg_aggregation(client_models, client_sizes)
    updated_control_variates = {}

    for key in global_model.keys():
       
        aggregated_variates = [
            control_variates[i].get(key, torch.zeros_like(global_model[key]).to(global_model[key].device))
            for i in range(len(client_models))
        ]
        updated_control_variates[key] = sum(aggregated_variates) / len(client_models)

    return global_model, updated_control_variates

# Example Usage
if __name__ == "__main__":
    # Example of dummy client models for aggregation
    client_models = [{"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])},
                     {"weight": torch.tensor([1.5, 2.5]), "bias": torch.tensor([0.7])}]
    client_sizes = [100, 200]

    aggregated_model = fedavg_aggregation(client_models, client_sizes)
    print("FedAvg Aggregated Model:", aggregated_model)
