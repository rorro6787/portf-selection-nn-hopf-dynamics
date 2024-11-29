# -*- coding: utf-8 -*-
"""
Author: Emilio Rodrigo Carreira Villalta
Date: 2024-11-29
Description: This script contains the PortfolioSelectionNN class, which models portfolio selection using a Hopfield-based neural network. The model aims to find an optimal portfolio by balancing risk and return, subject to cardinality and bounding constraints.
"""

import torch
import numpy as np

class PortfolioSelectionNN:
    def __init__(self, N, K, cov_matrix, mean_returns, epsilon, delta, lambda_val):
        """
        Initialize the Hopfield-based portfolio selection model.

        Args:
        - N: Number of assets in the portfolio.
        - K: Desired number of assets to include in the portfolio (cardinality constraint).
        - cov_matrix: Covariance matrix of asset returns, defining risk relationships between assets.
        - mean_returns: Mean return values for each asset.
        - epsilon: Lower bounds for the proportion of capital to be invested in each asset.
        - delta: Upper bounds for the proportion of capital to be invested in each asset.
        - lambda_val: Risk aversion parameter; 0 means prioritizing return, 1 means minimizing risk.
        """

        self.N = N  # Number of assets
        self.K = K  # Number of assets to include in the portfolio
        self.cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32)  # Risk (covariance) matrix
        self.mean_returns = torch.tensor(mean_returns, dtype=torch.float32)  # Expected returns
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)  # Lower bound constraints
        self.delta = torch.tensor(delta, dtype=torch.float32)  # Upper bound constraints
        self.lambda_val = lambda_val  # Risk aversion parameter
        
        # Calculate Hopfield network weights (representing relationships between assets)
        self.weights = -2 * lambda_val * self.cov_matrix  # Maps risk to weights
        # Bias term favors assets with higher returns
        self.bias = (1 - lambda_val) * self.mean_returns
    
    def activation_function(self, x, beta=1):
        """
        Sigmoid activation function for neuron states, bounded by epsilon and delta.

        Args:
        - x: Neuron state vector (portfolio weights).
        - beta: Gain parameter that controls the steepness of the sigmoid.
        
        Returns:
        - Activated neuron states, ensuring proportions lie within [epsilon, delta].
        """

        return self.epsilon + (self.delta - self.epsilon) / (1 + torch.exp(-beta * x))

    def energy_function(self, x):
        """
        Calculate the Hopfield energy function for a portfolio.
        Lower energy indicates a better portfolio.

        Args:
        - x: Current portfolio state (proportions of capital in each asset).
        
        Returns:
        - Energy value (objective function value).
        """

        energy = -0.5 * torch.sum(x @ self.weights * x) + torch.sum(self.bias * x)
        return energy

    def hopfield_dynamics(self, x, beta, num_iterations=100):
        """
        Update neuron states using Hopfield network dynamics to minimize energy.

        Args:
        - x: Initial state of the portfolio (randomly initialized proportions).
        - beta: Gain parameter for the sigmoid function (controls network convergence).
        - num_iterations: Number of update steps to perform.

        Returns:
        - Final state of the portfolio after network convergence.
        """

        for _ in range(num_iterations):
            # Update neuron states based on input weights, biases, and sigmoid activation
            x = self.activation_function(x + torch.mm(self.weights, x.unsqueeze(1)).squeeze() + self.bias, beta)
        return x
    
    def prune_neurons(self, x):
        """
        Enforce the cardinality constraint (limit number of assets to K) by pruning the smallest weights.

        Args:
        - x: Current portfolio state (proportions of capital).

        Returns:
        - Pruned portfolio state, where only K assets are retained.
        """

        # Select the top K assets with the highest proportions
        _, indices = torch.topk(x, self.K, largest=True)
        
        # Create a mask to zero out non-selected assets
        mask = torch.zeros_like(x)
        mask[indices] = 1  # Mark the top K assets
        
        # Apply the mask to retain only the top K assets
        return x * mask

    def greedy_constraint_adjustment(self, x):
        """
        Adjust the portfolio proportions to satisfy bounding and sum-to-one constraints.

        Args:
        - x: Current portfolio state.

        Returns:
        - Adjusted portfolio state satisfying all constraints.
        """

        # Ensure the portfolio weights are within the specified bounds [epsilon, delta]
        x = torch.clamp(x, self.epsilon, self.delta)

        # If the total weight exceeds 1, redistribute the excess
        if x.sum() > 1:
            excess = x.sum() - 1
            x -= excess / self.K  # Proportionally reduce the excess weight
        
        # Reapply bounds after redistribution
        return torch.clamp(x, self.epsilon, self.delta)

    def trace_efficient_frontier(self, lambda_values, num_portfolios):
        """
        Trace the efficient frontier by solving the optimization problem for different risk-return trade-offs.

        Args:
        - lambda_values: Range of risk aversion parameters (trade-off between return and risk).
        - num_portfolios: Number of random portfolios to evaluate per lambda value.

        Returns:
        - efficient_frontier: List of optimal portfolios for each lambda value.
        """
        efficient_frontier = []
        for lambda_val in lambda_values:
            # Update network parameters for the current risk-return trade-off
            self.lambda_val = lambda_val
            self.weights = -2 * lambda_val * self.cov_matrix
            self.bias = (1 - lambda_val) * self.mean_returns
            
            # Generate portfolios for the current lambda value
            portfolios = []
            for _ in range(num_portfolios):
                # Start with a random portfolio
                x = torch.rand(self.N)
                
                # Minimize energy using Hopfield dynamics
                x = self.hopfield_dynamics(x, beta=1)
                
                # Prune to enforce cardinality constraint
                x = self.prune_neurons(x)
                
                # Adjust to meet bounding and sum-to-one constraints
                x = self.greedy_constraint_adjustment(x)
                
                # Store the portfolio and its energy value
                portfolios.append((x, self.energy_function(x)))
            
            # Select the portfolio with the minimum energy
            efficient_frontier.append(min(portfolios, key=lambda p: p[1]))
        return efficient_frontier

# Main flow of the script
def main():
    # Example data setup
    N = 31  # Number of assets
    K = 10  # Desired number of assets in the portfolio
    cov_matrix = np.random.rand(N, N)  # Covariance matrix (example; replace with real data)
    mean_returns = np.random.rand(N)  # Mean returns for assets
    epsilon = 0.01 * np.ones(N)  # Lower bounds for investment proportions
    delta = np.ones(N)  # Upper bounds for investment proportions
    lambda_values = np.linspace(0, 1, 20)  # Risk aversion parameters (0: max return, 1: min risk)

    # Instantiate and solve
    model = PortfolioSelectionNN(N, K, cov_matrix, mean_returns, epsilon, delta, lambda_val=0)
    efficient_frontier = model.trace_efficient_frontier(lambda_values, num_portfolios=50)

    # Display results with enumeration
    print("Efficient Frontier Portfolios:")
    print("--------------------------------------------------------------------------")
    for idx, (portfolio, energy) in enumerate(efficient_frontier):
        print(f"Portfolio {idx + 1}:\n{portfolio.numpy()}\nEnergy: {energy.item()}")
        print("--------------------------------------------------------------------------")

# Execute the main flow
if __name__ == "__main__":
    main()