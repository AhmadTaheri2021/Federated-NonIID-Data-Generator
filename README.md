# Federated Non-IID Data Generator

This repository provides Python scripts and examples for generating synthetic non-IID (non-Identically Independently Distributed) datasets suitable for federated learning simulations. The code allows users to generate datasets with customizable heterogeneity levels by adjusting the alpha parameter.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage Example](#usage-example)
- [Explanation of Parameters](#explanation-of-parameters)
- [Visualization Examples](#visualization-examples)
- [License](#license)

## Overview

Federated learning scenarios often require datasets distributed unevenly (non-IID) across multiple clients. This repository provides an easy-to-use solution to create synthetic datasets with varying heterogeneity levels, allowing users to simulate and evaluate federated learning algorithms under realistic data distribution conditions.

## Installation

To install and set up the repository, follow these steps:

```bash
git clone https://github.com/your_username/Federated-NonIID-Data-Generator.git
cd Federated-NonIID-Data-Generator
pip install -r requirements.txt
```

### Requirements

- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`
- `jupyter`

## Usage Example

An illustrative example is provided in the notebook located at:

```
examples/usage_example.ipynb
```

Here's a brief example of usage:

```python
import numpy as np
from data_generator.data_partitioning import generate_distributed_datasets

# Example dataset
X_train = np.random.rand(1000, 20)
Y_train = np.random.randint(0, 10, 1000)

# Define client names and alpha values
client_names = [f'Client_{i}' for i in range(5)]
alpha_values = [0.01, 0.1, 1.0]

# Generate datasets
datasets = generate_distributed_datasets(
    X_train,
    Y_train,
    num_of_clients=5,
    client_names=client_names,
    alpha_values=alpha_values,
    show_dist=True
)
```

## Explanation of Parameters

- **`X_train`**, **`Y_train`**: Input training dataset.
- **`num_of_clients`**: Number of federated learning clients.
- **`client_names`**: List of client identifiers.
- **`alpha_values`**: Controls the degree of non-IID distribution:
  - Lower `alpha`: More heterogeneous data.
  - Higher `alpha`: More homogeneous data.
- **`show_dist`**: Enables visualization of the generated distributions.

## Visualization Examples

The function provides visualization tools to quickly assess how data distribution varies across different clients and alpha settings:

- **Bar plots**: Display the number of samples allocated per client.
- **Heatmaps**: Visualize the distribution of different classes across clients.

See the example notebook for detailed visualizations.

## License

Distributed under the MIT License. See the `LICENSE` file for more information.
