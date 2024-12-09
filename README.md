
# Federated Learning Aggregation Strategies Comparison

This repository is dedicated to benchmarking and comparing the performance of various aggregation strategies in federated learning, including **FedAvg**, **FedProx**, **FedNova**, and **SCAFFOLD**. The framework is designed to be modular, extensible, and easy to use for research and experimentation.

---

## **Features**

- **Aggregation Strategies**:
  - **FedAvg**: Simple weight averaging.
  - **FedProx**: Adds a proximal term to address client heterogeneity.
  - **FedNova**: Normalizes updates based on client contributions.
  - **SCAFFOLD**: Reduces variance using control variates.

- **Data Distribution Options**:
  - IID (Independent and Identically Distributed).
  - Non-IID (Dirichlet distribution).

- **Performance Tracking**:
  - Accuracy and loss trends over federated learning rounds.
  - Easy-to-generate plots for comparison.

- **Modular Components**:
  - Clients, aggregation strategies, data partitioning, and training logic are implemented as independent modules.

---

## **Installation**

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure PyTorch is installed. Visit [PyTorch.org](https://pytorch.org/) for installation instructions suitable for your platform.

---

## **Usage**

### **Running Experiments**

1. Configure experiment parameters directly in `main.py`:
   - Number of clients.
   - Aggregation strategy (e.g., FedAvg, FedProx).
   - Data distribution (IID or Dirichlet).
   - Other hyperparameters (e.g., learning rate, number of rounds).

2. Run the main script:
   ```bash
   python main.py
   ```

### **Results Visualization**

- Use `visualization.py` to generate accuracy and loss plots:
  ```bash
  python visualization.py
  ```
- Results are saved in the `results.json` file and visualizations in `.png` format.

---

## **Structure**

```plaintext
federated-learning-aggregation/
├── data_preparation.py     # Handles data loading and partitioning
├── model.py                # Defines model architecture (e.g., ResNet18)
├── client.py               # Implements client-side training logic
├── validator.py            # Tracks validation metrics
├── aggregation.py          # Defines aggregation strategies
├── main.py                 # Orchestrates federated learning rounds
├── visualization.py        # Generates result visualizations
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## **Aggregation Strategies**

### **FedAvg**
- Simple averaging of client weights.

### **FedProx**
- Adds a proximal term to prevent significant divergence of client updates.
- Suitable for heterogeneous environments.

### **FedNova**
- Normalizes client contributions to ensure fair aggregation.
- Accounts for varying amounts of client data or local training effort.

### **SCAFFOLD**
- Introduces control variates to reduce variance in client updates.
- Achieves faster convergence under non-IID settings.

---

## **Experiments**

### **Goal**
- Compare the performance of aggregation strategies under different settings:
  - Dataset: CIFAR-10, MNIST.
  - Data Distribution: IID and Dirichlet non-IID (α = 0.1, 0.5, 1.0).
  - Number of Clients: 10.
  - Rounds: 50.

### **Metrics**
- Global model accuracy and loss.
- Training time per round.
- Convergence rate across strategies.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.

