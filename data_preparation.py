# data_preparation.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def load_dataset(name, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if name == "MNIST":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif name == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataset, test_loader

def partition_data_iid(dataset, num_clients):
    total_samples = len(dataset)
    indices = np.random.permutation(total_samples)
    split_indices = np.array_split(indices, num_clients)
    return [Subset(dataset, idx) for idx in split_indices]

def partition_data_dirichlet(dataset, num_clients, alpha, seed=42):
    """
    Partition dataset into non-IID subsets using Dirichlet distribution.

    Parameters:
        dataset (Dataset): PyTorch dataset to partition.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet concentration parameter.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list[Subset]: List of PyTorch Subset objects, one for each client.
    """
    # Set random seed
    np.random.seed(seed)

    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        # Sample proportions for each client
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.array(proportions) * len(class_indices[c])).astype(int)

        # Shuffle indices
        indices = np.random.permutation(class_indices[c])
        start = 0
        for client_id, proportion in enumerate(proportions):
            client_indices[client_id].extend(indices[start:start + proportion])
            start += proportion

    return [Subset(dataset, idx) for idx in client_indices]

def create_data_loaders(partitioned_data, batch_size):
    return [DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True) for subset in partitioned_data]

# Example Usage
if __name__ == "__main__":
    dataset, test_loader = load_dataset("CIFAR10", batch_size=32)
    iid_data = partition_data_iid(dataset, num_clients=10)
    non_iid_data = partition_data_dirichlet(dataset, num_clients=10, alpha=0.5)
    iid_loaders = create_data_loaders(iid_data, batch_size=32)
    non_iid_loaders = create_data_loaders(non_iid_data, batch_size=32)
