import matplotlib.pyplot as plt

'''
File with functions related to plots
'''

def plot_strategy_w_costs(results, transaction_cost):
    plt.figure(figsize=(12, 8))

    target_performance_cumulative_returns = results["target_performance_cumulative_returns"].iloc[0]
    cumulative_returns_w_optimal_costs = results["performance_cumulative_returns_w_optimal_costs"].iloc[0]

    plt.plot(cumulative_returns_w_optimal_costs.index, cumulative_returns_w_optimal_costs.values, label="Strategy with Optimal Costs")
    plt.plot(target_performance_cumulative_returns.index, target_performance_cumulative_returns.values, label="Target Strategy")

    plt.title(f"Strategy Cumulative Returns with a Transaction Cost of: {transaction_cost}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()

def plot_strategy_w_costs_and_benchmark(results, transaction_cost):
    plt.figure(figsize=(12, 8))

    target_performance_cumulative_returns = results["target_performance_cumulative_returns"].iloc[0]
    cumulative_returns_w_optimal_costs = results["performance_cumulative_returns_w_optimal_costs"].iloc[0]
    benchmark_cumulative_returns = results["benchmark_cumulative_returns"]

    plt.plot(cumulative_returns_w_optimal_costs.index, cumulative_returns_w_optimal_costs.values, label="Strategy with Optimal Costs")
    plt.plot(target_performance_cumulative_returns.index, target_performance_cumulative_returns.values, label="Target Strategy")
    plt.plot(cumulative_returns_w_optimal_costs.index, benchmark_cumulative_returns.values, label="Benchmark")

    plt.title(f"Strategy Cumulative Returns with a Transaction Cost of: {transaction_cost}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()

