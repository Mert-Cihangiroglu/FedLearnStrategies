# validator.py
import torch

class Validator:
    """
    A class for validating the global model during federated learning.

    Attributes:
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations (e.g., 'cpu', 'mps' or 'cuda').
    """
    def __init__(self, test_loader, device):
        self.test_loader = test_loader
        self.device = device

    def validate(self, model):
        """
        Validate the global model on the test dataset.

        Parameters:
            model (torch.nn.Module): The global model to be validated.

        Returns:
            dict: A dictionary containing validation accuracy and loss.
        """
        model.to(self.device)
        model.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)

        return {"accuracy": accuracy, "loss": avg_loss}

    def log_results(self, metrics, round_number):
        """
        Log the validation metrics.

        Parameters:
            metrics (dict): Dictionary containing accuracy and loss.
            round_number (int): The current federated learning round.
        """
        print(f"Round {round_number}: Accuracy = {metrics['accuracy']:.2f}%, Loss = {metrics['loss']:.4f}")

# Example Usage
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from model import get_model

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    validator = Validator(test_loader=test_loader, device=torch.device("cpu"))
    model = get_model(output_classes=10)

    metrics = validator.validate(model)
    validator.log_results(metrics, round_number=1)
