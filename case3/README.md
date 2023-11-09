# Portfolio Allocation Strategies

This case3 contains multiple strategies for allocating a financial portfolio based on historical price data. The goal of each strategy is to maximize the Sharpe ratio, which is a measure of risk-adjusted return. The different strategies employ various optimization techniques and statistical measures to determine the best allocation of assets.

## Strategies Overview

### Genetic Algorithm Approach (`GeneticAlgorithm` class)
- Uses a genetic algorithm to optimize portfolio weights based on historical returns.
- Evaluates the fitness of a portfolio by its Sharpe ratio.
- Employs operations like crossover and mutation to evolve the population of portfolio weights.
- Determines convergence when the Sharpe ratio stops improving significantly.

### Maximize Sharpe Ratio (`MaximizeSharpeModel` class)
- Directly maximizes the Sharpe ratio by adjusting portfolio weights.
- Takes into account the mean returns, covariance, skewness, and kurtosis of asset returns.
- Utilizes the `scipy.optimize.minimize` function to find the optimal weights.

### Markowitz Portfolio with Conditional Value at Risk (CVaR) (`markowitz_cvar_portfolio` function)
- Enhances the traditional Markowitz portfolio optimization by incorporating CVaR.
- CVaR is used as a risk measure instead of variance to account for tail risk.
- Uses the historical mean, covariance, skewness, and kurtosis to calculate CVaR.
- Optimizes the portfolio allocation to minimize CVaR while considering constraints on the weights.

### Projections (`Projections` class)
- Generates projected returns based on historical data using the Laplace distribution.
- Simulates possible future return paths for assets in the portfolio.

## Portfolio Allocation Process

1. Historical price data is loaded and processed.
2. Different strategies are applied to allocate the portfolio:
   - Genetic algorithm is run over multiple generations to find the optimal weights.
   - Sharpe ratio maximization is performed for a direct optimization approach.
   - CVaR optimization is used to allocate the portfolio considering extreme risk scenarios.
3. Portfolio weights are normalized to ensure the sum is equal to 1 (100% of the capital is allocated).
4. Portfolio performance is evaluated based on the backtested Sharpe ratio.

## Usage

To use these strategies, follow these steps:

1. Load your historical price data into a DataFrame.
2. Pass the data to the strategy class or function of your choice.
3. Call the appropriate methods to perform the optimization.
4. Retrieve the optimal weights and apply them to your portfolio.

## Grading Function (`grading` function)

- The `grading` function is used to evaluate the performance of the portfolio allocation strategies.
- It simulates the allocation process over historical data and calculates the final Sharpe ratio.
- The function returns the Sharpe ratio, the evolution of the capital over time, and the weights used each day.

## Requirements
- Python 3
- NumPy
- Pandas
- SciPy
- Install the required packages using pip:

`pip install numpy pandas scipy`


## Results

The strategies are compared based on the Sharpe ratio they achieve. Higher Sharpe ratios indicate better risk-adjusted returns.

## Conclusion

By utilizing different optimization techniques and statistical measures, these strategies aim to provide robust methods for portfolio allocation in various market conditions.

