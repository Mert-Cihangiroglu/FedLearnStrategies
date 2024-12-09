
# Federated Learning Aggregation Strategies Benchmark

This repository is part of my research project aimed at benchmarking federated learning aggregation strategies, specifically **FedAvg**, **FedProx**, **FedNova**, and **SCAFFOLD**. The goal is to compare their performance under different conditions and evaluate their suitability for various real-world scenarios.

---

## Features

### Aggregation Strategies
- **FedAvg**: Basic weight averaging.
- **FedProx**: Adds a proximal term to tackle client heterogeneity.
- **FedNova**: Normalizes updates to balance client contributions.
- **SCAFFOLD**: Reduces variance using control variates.

### Data Distribution
- **IID**: Data is evenly distributed among clients.
- **Non-IID**: Data is distributed non-uniformly using Dirichlet distribution.

### Performance Tracking
- Tracks global model accuracy and loss over multiple federated rounds.
- Produces visualizations to compare the strategies.

### Modular Design
Each component is modular and easy to adapt:
- Clients, data preparation, aggregation methods, and training logic are implemented separately for flexibility.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/federated-learning-benchmark.git
   cd federated-learning-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure PyTorch is installed. You can get installation instructions specific to your system from [PyTorch.org](https://pytorch.org/).

---

## Usage

### Preparing Tiny ImageNet
For experiments involving Tiny ImageNet, ensure the dataset is prepared by running:
```bash
python tiny-image-net-downloader.py
```
After running this script, place the generated `tiny-224` folder under the `data` directory.

### Running Experiments

1. Modify the `experiments` list in `main.py` to define:
   - The dataset (e.g., `MNIST`, `CIFAR10`, `CIFAR100`, `TinyImageNet`).
   - Data distribution (`IID` or `Dirichlet`).
   - Number of clients, aggregation strategy, and other parameters.

2. Run the main script to start the experiment:
   ```bash
   python main.py
   ```

### Visualizing Results

1. Use `visualization.py` to generate plots:
   ```bash
   python visualization.py
   ```

2. Results are saved as JSON files and plots in the `results/` directory.

---

## Project Structure

```plaintext
federated-learning-benchmark/
├── data_preparation.py     # Prepares datasets and partitions data
├── model.py                # Defines model architecture (e.g., ResNet18)
├── client.py               # Implements client-side training
├── validator.py            # Tracks and logs validation metrics
├── aggregation.py          # Implements aggregation strategies
├── main.py                 # Orchestrates the experiments
├── visualization.py        # Generates visualizations for results
├── requirements.txt        # Python dependencies
├── tiny-image-net-downloader.py # Downloads and prepares Tiny ImageNet
└── README.md               # Project overview
```

---

## Aggregation Strategies Overview

### FedAvg
Averages client updates without considering heterogeneity. Simple and effective for homogeneous setups.

### FedProx
Adds a regularization term to reduce update divergence among clients, making it more robust in heterogeneous environments.

### FedNova
Balances updates by normalizing them based on client contributions, addressing imbalances in data and computational power.

### SCAFFOLD
Uses control variates to correct updates and reduce variance, especially under non-IID distributions.

---

## Experiments

### Objective
To test and benchmark the aggregation strategies under the following conditions:

- **Datasets**: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet.
- **Data Distributions**: IID and Dirichlet non-IID (α = 0.1, 0.5, 1.0).
- **Clients**: 10.
- **Federated Rounds**: 50.

### Metrics
- Model accuracy and loss.
- Training time per round.
- Convergence rate for each strategy.

---

## License
This repository is licensed under the MIT License. See the `LICENSE` file for details.
