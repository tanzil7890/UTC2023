# University Of Chicago Trading Competition 2023

# Financial Portfolio Optimization


This project encompasses a suite of financial analysis tools designed to perform various tasks such as optimizing portfolio weights using genetic algorithms, evaluating risk with Conditional Value at Risk (CVaR), and maximizing the Sharpe ratio to achieve the best risk-adjusted returns.

## Overview

The project is divided into several Jupyter Notebooks and Python scripts, each fulfilling a specific role in financial data analysis and portfolio management. Key components include:

- **Genetic Algorithm Implementation**: A Python class that implements a genetic algorithm to optimize portfolio weights based on historical returns, aiming to maximize the Sharpe ratio.
- **Markowitz Portfolio with CVaR**: An approach that extends the traditional Markowitz portfolio optimization by incorporating CVaR as a measure of risk.
- **Sharpe Ratio Maximization**: Models and scripts designed to maximize the Sharpe ratio, considering factors such as skewness and kurtosis of asset returns.
- **Data Generation**: Utilities for generating projected financial data, simulating various market conditions for testing optimization algorithms.

## Features

- **Portfolio Optimization**: Utilize historical data to compute optimal asset weights that maximize the Sharpe ratio while considering higher moments of distribution like skewness and kurtosis.
- **Risk Assessment with CVaR**: Assess and minimize portfolio risk using CVaR, a risk measure that focuses on the tail end of the distribution of returns.
- **Algorithmic Trading Simulation**: Test trading strategies using generated data to simulate different market scenarios and evaluate performance across multiple runs.

## Installation

To set up the project, ensure you have a Python environment with the necessary libraries installed, including `pandas`, `numpy`, `scipy`, and `matplotlib`.

```bash
pip install pandas numpy scipy matplotlib
```

## Usage
To run the optimization models and simulations:

1. Load the financial data into the provided Python scripts or Jupyter Notebooks.
2. Choose the appropriate model or algorithm you wish to run.
3. Execute the script or notebook cells in sequence to perform the analysis.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request for review.