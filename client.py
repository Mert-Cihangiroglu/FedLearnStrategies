
# client.py
import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    """
    A class representing a client in federated learning.
    
    Attributes:
        client_id (int): Unique identifier for the client.
        data_loader (torch.utils.data.DataLoader): Data loader for the client's local data.
        device (torch.device): Device to perform computations (e.g., 'cpu' or 'cuda').
    """
    def __init__(self, client_id, data_loader, device):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.control_variates = {}  # For SCAFFOLD

    def local_train(self, model, epochs, criterion, optimizer, strategy_params=None):
        """
        Perform local training on the client's data.

        Parameters:
            model (torch.nn.Module): The model to train.
            epochs (int): Number of training epochs.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            strategy_params (dict, optional): Additional parameters for specific strategies (e.g., FedProx, SCAFFOLD).

        Returns:
            dict: The updated model state after training.
        """
        model.to(self.device)
        model.train()
        
        #CHECK IF Mu exist, if it doesn't set it to 0.
        mu = strategy_params.get("mu", 0.0) if strategy_params else 0.0
        global_control_variates = strategy_params.get("global_control_variates", None)

        # Initialize control variates for SCAFFOLD if needed
        if global_control_variates and not self.control_variates:
            self.control_variates = {k: torch.zeros_like(v).to(self.device) for k, v in global_control_variates.items()}

        global_params = {name: param.clone().detach() for name, param in model.state_dict().items()}

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)

                # FedProx-specific proximal term
                if mu > 0.0:
                    prox_loss = 0.0
                    for name, param in model.state_dict().items():
                        prox_loss += ((param - global_params[name]) ** 2).sum()
                    loss += (mu / 2) * prox_loss

                # SCAFFOLD-specific adjustments
                if global_control_variates:
                    scaffold_adjustment = 0.0
                    for name, param in model.named_parameters():
                        scaffold_adjustment += torch.sum(
                            (self.control_variates[name] - global_control_variates[name]) * param
                        )
                    loss += scaffold_adjustment

                loss.backward()
                optimizer.step()

        # Update local control variates for SCAFFOLD
        if global_control_variates:
            for name, param in model.named_parameters():
                self.control_variates[name] += param.grad.data.clone()

        return model.state_dict()

# Example Usage
if __name__ == "__main__":
    # Dummy dataset and model for demonstration
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from model import get_model

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    client = Client(client_id=1, data_loader=data_loader, device=torch.device("cpu"))
    model = get_model(output_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    updated_model_state = client.local_train(model, epochs=1, criterion=criterion, optimizer=optimizer)
