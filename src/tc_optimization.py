import enum
import numpy as np
import pandas as pd

class DistanceMethod(str, enum.Enum):
    """
    Enum representing different distance calculation methods.
    """
    PEARSON = "Pearson"
    MANHATTAN = "Manhattan"
    NORMALIZED_EUCLIDEAN = "Normalized Euclidean"
    EUCLIDEAN = "Euclidean"
    ONE_MINUS_CORRELATION = "one-minus-correlation"
    CANBERRA = "Canberra"
    ONE_MINUS_COSINE = "one-minus-cosine"
    BRAY_CURTIS = "Bray-Curtis"

    @staticmethod
    def calculate_distance(method: 'DistanceMethod', w_target: np.array, w_unbalanced: np.array) -> float:
        """
        Calculate the distance between unbalanced weights and target weights.

        Parameters:
            method (DistanceMethod): The method to use for calculating the distance.
            w_target (np.array): The target weights defined by the strategy.
            w_unbalanced (np.array): The weights at the end of the rebalancing period.

        Returns:
            float: The distance between the unbalanced and target weights.
        """
        if method == DistanceMethod.PEARSON:
            return np.corrcoef(w_target, w_unbalanced)[0, 1]
        elif method == DistanceMethod.MANHATTAN:
            return np.sum(np.abs(w_target - w_unbalanced))
        elif method == DistanceMethod.NORMALIZED_EUCLIDEAN:
            return 1 / np.sqrt(len(w_target)) * np.linalg.norm(w_target - w_unbalanced)
        elif method == DistanceMethod.EUCLIDEAN:
            return np.linalg.norm(w_target - w_unbalanced)
        elif method == DistanceMethod.ONE_MINUS_CORRELATION:
            return 1 - np.dot(w_target - np.mean(w_target), w_unbalanced - np.mean(w_unbalanced)) / (
                np.linalg.norm(w_target - np.mean(w_target)) * np.linalg.norm(w_unbalanced - np.mean(w_unbalanced))
            )
        elif method == DistanceMethod.CANBERRA:
            return np.sum(np.abs(w_target - w_unbalanced) / (np.abs(w_target) + np.abs(w_unbalanced)))
        elif method == DistanceMethod.ONE_MINUS_COSINE:
            return 1 - np.dot(w_target, w_unbalanced) / (np.linalg.norm(w_target) * np.linalg.norm(w_unbalanced))
        elif method == DistanceMethod.BRAY_CURTIS:
            return np.sum(np.abs(w_target - w_unbalanced)) / (np.sum(np.abs(w_target)) + np.sum(np.abs(w_unbalanced)))
        else:
            raise ValueError("Invalid distance method")

def adjust_alpha(method, w_target, w_unbalanced, delta, verbose=False, tomas=False):
    """
    Adjust alpha based on delta and the distance between weights.

    Parameters:
        method (str): The method to use for calculating the distance.
        w_target (np.array): The target weights defined by the strategy.
        w_unbalanced (np.array): The weights at the end of the rebalancing period.
        delta (float): The threshold for adjusting alpha.
        verbose (bool): Whether to print additional information.
        tomas (bool): formato dos pesos: True para [[ticker1, data, peso1], ...] e False para DataFrame (colunas assets, linhas datas). Lore: homenagem ao PM de 

    Returns:
        float: The alpha value (proportion of the target weights for the next rebalancing period).
    """

    if tomas:
        # transformar os pesos no nosso formato

        w_target = pd.DataFrame(w_target, columns=['ticker', 'Date', 'weight'])
        w_unbalanced = pd.DataFrame(w_unbalanced, columns=['ticker', 'Date', 'weight'])


        w_target = w_target.pivot(index='Date', columns='ticker', values='weight')
        w_unbalanced = w_unbalanced.pivot(index='Date', columns='ticker', values='weight')

        w_target = w_target.reset_index()
        w_unbalanced = w_unbalanced.reset_index()
    
    distance = DistanceMethod.calculate_distance(method, w_target, w_unbalanced)
    if distance == 0:
        distance = 1e-10 # Avoid division by zero
    
    if verbose:
        print(f"Distance ({method}): {distance}")
        print(f"Delta: {delta}")

    """ if distance > delta:
        alpha = max(0, delta/distance)
    else:
        alpha = 1 """

    print(f"Distance between {w_target} and {w_unbalanced} is {distance}")
    return min(1, max(0, delta / distance)) if distance > delta else 1





def optimize_alpha(w_target, w_unbalanced, trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date, gamma = 3, alpha_range = (0, 1), step = 0.05, transaction_cost = 0.01, verbose=False):
    """
    Optimize alpha to maximize the Constant Relative Risk Aversion (CRRA).

    Parameters:
        w_target (np.array): The target weights defined by the strategy.
        w_unbalanced (np.array): The weights at the end of the rebalancing period.
        returns (np.array): The returns of the assets.
        gamma (float): The risk aversion parameter (CRRA utility).
        alpha_range (tuple): The range of alpha values to test.
        step (float): The step size for alpha.
        transaction_cost (float): Transaction costs for rebalancing.
        verbose (bool): Whether to print additional information.

    Returns:
        tuple: The optimized alpha value, its corresponding CRRA and the optimal delta.
    """
    if alpha_range[0] >= alpha_range[1]:
        raise ValueError("Invalid alpha_range: start must be less than end.")

    best_alpha = None
    best_crra = -np.inf

    # Calcular os retornos futuros dos ativos
    future_assets_returns = get_future_assets_returns(trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date)

    for alpha in np.arange(alpha_range[0], alpha_range[1], step):
        w_rebalanced = alpha * w_unbalanced + (1 - alpha) * w_target

        available_assets = future_assets_returns.columns.tolist()
        asset_names = previous_dataframe_assets.columns  # Full list of assets
        weights_dict = dict(zip(asset_names, w_rebalanced))  # Map asset → weight
        filtered_weights = np.array([weights_dict[asset] for asset in available_assets if asset in weights_dict])

        # Select only the returns for the available assets
        filtered_returns = future_assets_returns.iloc[0, :].values  # Only the first row of returns

        #print(f"Parameters used to calculuate portfolio returns: w_rebalanced: {filtered_weights}, returns: {filtered_returns}, alpha_t: {alpha_t}")
        portfolio_returns_before_costs = np.dot(filtered_weights, filtered_returns)
        portfolio_returns = portfolio_returns_before_costs - transaction_cost * np.sum(np.abs(w_rebalanced - w_unbalanced))

        if gamma == 1:
            crra = np.mean(np.log(1 + portfolio_returns))  # gamma = 1 (log utility)
        else:
            crra = np.mean((1 + portfolio_returns)**(1 - gamma) / (1 - gamma))

        if verbose:
            print(f"Alpha: {alpha}, Portfolio Returns: {portfolio_returns}, CRRA: {crra}")

        if (best_alpha is None or crra > best_crra or (crra == best_crra and alpha > best_alpha)):
            best_crra = crra
            best_alpha = alpha
            if verbose:
                print(f"New best alpha: {best_alpha}, CRRA: {best_crra}")

    return best_alpha, best_crra
    
def optimize_delta(w_target, w_unbalanced, returns, distance_method = "Euclidean", gamma = 3, delta_range = (-10, 10), step = 1, transaction_cost = 0.01, verbose=False):
    """
    Optimize delta to maximize the Constant Relative Risk Aversion (CRRA).

    Parameters:
        w_target (np.array): The target weights defined by the strategy.
        w_unbalanced (np.array): The weights at the end of the rebalancing period.
        returns (np.array): The returns of the assets.
        distance_method (str): The distance method to calculate alpha.
        gamma (float): The risk aversion parameter (CRRA utility).
        delta_range (tuple): The range of delta values to test.
        step (float): The step size for delta.
        transaction_cost (float): Transaction costs for rebalancing.
        verbose (bool): Whether to print additional information.

    Returns:
        tuple: The optimized delta value, its corresponding CRRA and the optimal alpha.
    """
    if delta_range[0] >= delta_range[1]:
        raise ValueError("Invalid delta_range: start must be less than end.")

    best_delta = None
    best_crra = -np.inf

    print(f"The delta optimization is using the following parameters: \n w_target: {w_target}, w_unbalanced: {w_unbalanced}, returns: {returns}, distance_method: {distance_method}, gamma: {gamma}, delta_range: {delta_range}, step: {step}, transaction_cost: {transaction_cost}, verbose: {verbose}")
    for delta in np.arange(delta_range[0], delta_range[1], step):
        
        alpha_t = adjust_alpha(method=distance_method, w_target=w_target, w_unbalanced=w_unbalanced, delta=delta)
            
        w_rebalanced = alpha_t * w_unbalanced + (1 - alpha_t) * w_target
        
        # Usar os retornos da proxima iteracao: para os MR usar a media e para trend usar o ultimo
        portfolio_returns_before_costs = np.dot(w_rebalanced, returns)

        portfolio_returns = portfolio_returns_before_costs - transaction_cost * np.sum(np.abs(w_rebalanced - w_unbalanced))

        if gamma == 1:
            crra = np.mean(np.log(1 + portfolio_returns))  # gamma = 1 (log utility)
        else:
            crra = np.mean((1 + portfolio_returns)**(1 - gamma) / (1 - gamma))
    
        #if verbose:
            #print(f"Delta: {delta}, Portfolio Returns: {portfolio_returns}, CRRA: {crra}")

        if (best_delta is None or crra > best_crra or (crra == best_crra and delta > best_delta)):
            if delta >= 0:  # Restrição adicional
                best_crra = crra
                best_delta = delta
                best_alpha = alpha_t
                #if verbose:
                    #print(f"New best delta: {best_delta}, CRRA: {best_crra}")



    return best_delta, best_crra, best_alpha

def calculate_crra(delta, w_target, w_unbalanced, trendy_assets_df, mean_reverting_assets_df, previous_date, date, previous_dataframe_assets, distance_method="Euclidean", gamma=3, transaction_cost=0.01):
    """
    Calculate the Constant Relative Risk Aversion (CRRA) for a given delta.

    Parameters:
        delta (float): The threshold for adjusting alpha.
        w_target (np.array): The target weights defined by the strategy.
        w_unbalanced (np.array): The weights at the end of the rebalancing period.
        future_assets_returns (pd.DataFrame): DataFrame containing the predicted assets returns in the next period.
        previous_dataframe_assets (pd.DataFrame): DataFrame with the assets returns in the previous period.
        distance_method (str): The distance method to calculate alpha.
        gamma (float): The risk aversion parameter (CRRA utility).
        transaction_cost (float): Transaction costs for rebalancing.
    """

    alpha_t = adjust_alpha(method=distance_method, w_target=w_target, w_unbalanced=w_unbalanced, delta=delta)
    w_rebalanced = alpha_t * w_unbalanced + (1 - alpha_t) * w_target

    # Calcular os retornos futuros dos ativos
    future_assets_returns = get_future_assets_returns(trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date)

    available_assets = future_assets_returns.columns.tolist()
    asset_names = previous_dataframe_assets.columns  # Full list of assets
    weights_dict = dict(zip(asset_names, w_rebalanced))  # Map asset - weight
    filtered_weights = np.array([weights_dict[asset] for asset in available_assets if asset in weights_dict])

    # Select only the returns for the available assets
    filtered_returns = future_assets_returns.iloc[0, :].values  # Only the first row of returns

    #print(f"Parameters used to calculuate portfolio returns: w_rebalanced: {filtered_weights}, returns: {filtered_returns}, alpha_t: {alpha_t}")
    portfolio_returns_before_costs = np.dot(filtered_weights, filtered_returns)
    portfolio_returns = portfolio_returns_before_costs - transaction_cost * np.sum(np.abs(w_rebalanced - w_unbalanced))
    print(f"Portfolio Returns: {portfolio_returns}")

    if gamma == 1:
        return np.log(1 + portfolio_returns), alpha_t  # gamma = 1 (log utility)
    else:
        return (1 + portfolio_returns)**(1 - gamma) / (1 - gamma), alpha_t

def optimize_delta_refined(w_target, w_unbalanced, trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date, distance_method="Euclidean", gamma=3, delta_range=(-10, 10), initial_step=1, transaction_cost=0.01, verbose=False):
    """
    Optimize delta to maximize the Constant Relative Risk Aversion (CRRA) using a grid search.

    Parameters:
        w_target (np.array): The target weights defined by the strategy.
        w_unbalanced (np.array): The weights at the end of the rebalancing period.
        trendy_assets_df (pd.DataFrame): DataFrame containing trendy assets with columns ['asset', 'date'].
        mean_reverting_assets_df (pd.DataFrame): DataFrame containing mean reverting assets with columns ['asset', 'date'].
        previous_dataframe_assets (pd.DataFrame): DataFrame with the assets returns in the previous period.
        previous_date (int): The previous date.
        date (int): The current date.
        distance_method (str): The distance method to calculate alpha.
        gamma (float): The risk aversion parameter (CRRA utility).
        delta_range (tuple): The range of delta values to test.
        initial_step (float): The initial step size for delta.
        transaction_cost (float): Transaction costs for rebalancing.
        verbose (bool): Whether to print additional information.

    Returns:
        tuple: The optimized delta value, its corresponding CRRA, and the optimal alpha.
    """
    # Validar entradas
    if delta_range[0] >= delta_range[1]:
        raise ValueError("Invalid delta_range: start must be less than end.")
    if w_target is None or w_unbalanced is None or previous_dataframe_assets is None:
        raise ValueError("Invalid inputs: w_target, w_unbalanced, and returns must be provided.")
    
    distance = DistanceMethod.calculate_distance(distance_method, w_target, w_unbalanced)
    distance = max(distance, 1e-10)  # Evitar divisão por zero
    delta_range = (0, distance)

    if verbose:
        print(f"Distance calculated: {distance}. Delta range set to [0, {distance}].")
    
    # Calcular os retornos futuros dos ativos
    future_assets_returns = get_future_assets_returns(trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date)

    #print(f"returns used as rt+1: {returns}")
    def calculate_crra(delta):
        #Helper function to calculate CRRA for a given delta.
        alpha_t = adjust_alpha(method=distance_method, w_target=w_target, w_unbalanced=w_unbalanced, delta=delta)
        w_rebalanced = alpha_t * w_unbalanced + (1 - alpha_t) * w_target

        available_assets = future_assets_returns.columns.tolist()
        asset_names = previous_dataframe_assets.columns  # Full list of assets
        weights_dict = dict(zip(asset_names, w_rebalanced))  # Map asset - weight
        filtered_weights = np.array([weights_dict[asset] for asset in available_assets if asset in weights_dict])

        # Select only the returns for the available assets
        filtered_returns = future_assets_returns.iloc[0, :].values  # Only the first row of returns

        #print(f"Parameters used to calculuate portfolio returns: w_rebalanced: {filtered_weights}, returns: {filtered_returns}, alpha_t: {alpha_t}")
        portfolio_returns_before_costs = np.dot(filtered_weights, filtered_returns)
        portfolio_returns = portfolio_returns_before_costs - transaction_cost * np.sum(np.abs(w_rebalanced - w_unbalanced))

        if gamma == 1:
            return np.mean(np.log(1 + portfolio_returns))  # gamma = 1 (log utility)
        else:
            return np.mean((1 + portfolio_returns)**(1 - gamma) / (1 - gamma))

    best_delta = None
    best_crra = -np.inf
    step = initial_step
    previous_best_delta = None

    while step >= initial_step/1000:
        if verbose:
            print(f"\n Refining search with step: {step}")

        delta_values = np.arange(delta_range[0], delta_range[1] + step, step)
        crra_values = np.array([calculate_crra(delta) for delta in delta_values])

        max_index = np.argmax(crra_values)
        current_best_delta = delta_values[max_index]
        current_best_crra = crra_values[max_index]

        if verbose:
            print(f"Best delta in this iteration: {current_best_delta:.4f}, CRRA: {current_best_crra:.6f}")

        if np.isclose(current_best_crra, best_crra) and current_best_delta == previous_best_delta:
            if verbose:
                print(" Converged: No improvement detected.")
            break
        
        best_delta = current_best_delta
        best_crra = current_best_crra
        previous_best_delta = best_delta

        # Refinar o intervalo de busca ao redor do melhor delta encontrado
        delta_range = (max(0, best_delta - step), min(distance, best_delta + step))
        step /= 10

    best_alpha = adjust_alpha(method=distance_method, w_target=w_target, w_unbalanced=w_unbalanced, delta=best_delta)

    if verbose:
        print(f"\n Final Results -> Optimized delta: {best_delta:.4f}, CRRA: {best_crra:.6f}, Alpha: {best_alpha:.4f}")

    return best_delta, best_crra, best_alpha

def optimize_delta_refined(w_target, w_unbalanced, trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date, distance_method="Euclidean", gamma=3, delta_range=(-10, 10), initial_step=1, transaction_cost=0.01, verbose=False):
    """
    Optimize delta to maximize the Constant Relative Risk Aversion (CRRA) using a grid search.

    Parameters:
        w_target (np.array): The target weights defined by the strategy.
        w_unbalanced (np.array): The weights at the end of the rebalancing period.
        trendy_assets_df (pd.DataFrame): DataFrame containing trendy assets with columns ['asset', 'date'].
        mean_reverting_assets_df (pd.DataFrame): DataFrame containing mean reverting assets with columns ['asset', 'date'].
        previous_dataframe_assets (pd.DataFrame): DataFrame with the assets returns in the previous period.
        previous_date (int): The previous date.
        date (int): The current date.
        distance_method (str): The distance method to calculate alpha.
        gamma (float): The risk aversion parameter (CRRA utility).
        delta_range (tuple): The range of delta values to test.
        initial_step (float): The initial step size for delta.
        transaction_cost (float): Transaction costs for rebalancing.
        verbose (bool): Whether to print additional information.

    Returns:
        tuple: The optimized delta value, its corresponding CRRA, and the optimal alpha.
    """
    # Validar entradas
    if delta_range[0] >= delta_range[1]:
        raise ValueError("Invalid delta_range: start must be less than end.")
    if w_target is None or w_unbalanced is None or previous_dataframe_assets is None:
        raise ValueError("Invalid inputs: w_target, w_unbalanced, and returns must be provided.")
    
    distance = DistanceMethod.calculate_distance(distance_method, w_target, w_unbalanced)
    distance = max(distance, 1e-10)  # Evitar divisão por zero
    delta_range = (0, distance)

    if verbose:
        print(f"Distance calculated: {distance}. Delta range set to [0, {distance}].")

    #print(f"returns used as rt+1: {returns}")

    best_delta = None
    best_crra = -np.inf
    step = initial_step
    previous_best_delta = None

    while step >= initial_step/1000:
        if verbose:
            print(f"\n Refining search with step: {step}")

        delta_values = np.arange(delta_range[0], delta_range[1] + step, step)
        crra_values = np.array([calculate_crra(delta) for delta in delta_values])

        max_index = np.argmax(crra_values)
        current_best_delta = delta_values[max_index]
        current_best_crra = crra_values[max_index]

        if verbose:
            print(f"Best delta in this iteration: {current_best_delta:.4f}, CRRA: {current_best_crra:.6f}")

        if np.isclose(current_best_crra, best_crra) and current_best_delta == previous_best_delta:
            if verbose:
                print(" Converged: No improvement detected.")
            break
        
        best_delta = current_best_delta
        best_crra = current_best_crra
        previous_best_delta = best_delta

        # Refinar o intervalo de busca ao redor do melhor delta encontrado
        delta_range = (max(0, best_delta - step), min(distance, best_delta + step))
        step /= 10

    best_alpha = adjust_alpha(method=distance_method, w_target=w_target, w_unbalanced=w_unbalanced, delta=best_delta)

    if verbose:
        print(f"\n Final Results -> Optimized delta: {best_delta:.4f}, CRRA: {best_crra:.6f}, Alpha: {best_alpha:.4f}")

    return best_delta, best_crra, best_alpha


def apply_transaction_cost(returns, transaction_cost, weights, previous_weights):
    """
    Apply transaction cost to the returns based on the change in weights.

    Parameters:
        returns (np.array or float): The returns for the period.
        transaction_cost (float): The transaction cost rate (e.g., 0.1 for 10%).
        weights (np.array): The current weights of the portfolio.
        previous_weights (np.array): The weights from the previous period. If None, no transaction cost is applied.

    Returns:
        np.array or float: The returns after applying transaction costs.
        cost_of_transaction (float): The total cost of transactions based on the weight changes.
    """
    weights = np.array(weights)
    previous_weights = np.array(previous_weights)
    weight_changes = np.abs(weights - previous_weights)
    print(f"\n Weights: {weights}, previous_weights: {previous_weights}, weight_changes: {weight_changes}")
    cost_of_transaction = transaction_cost * np.sum(weight_changes)
    print(f" Cost of transaction: {cost_of_transaction}")

    return returns - transaction_cost * np.sum(weight_changes), cost_of_transaction

def get_future_assets_returns(trendy_assets_df, mean_reverting_assets_df, previous_dataframe_assets, previous_date, date):
    """
    Get the future returns of the assets based on the signals.

    Parameters:
    trendy_assets_df (pd.DataFrame): DataFrame containing trendy assets with columns ['asset', 'date'].
    mean_reverting_assets_df (pd.DataFrame): DataFrame containing mean reverting assets with columns ['asset', 'date'].
    previous_dataframe_assets (pd.DataFrame): DataFrame with the assets returns in the previous period.
    previous_date (int): The previous date.
    date (int): The current date.

    Returns:
    pd.DataFrame: DataFrame containing the predicted assets returns in the next period.
    """
    trendy_assets_prev = trendy_assets_df.loc[trendy_assets_df['date'] == previous_date, 'asset'].tolist()
    trendy_assets_curr = trendy_assets_df.loc[trendy_assets_df['date'] == date, 'asset'].tolist()
    mean_reverting_assets_prev = mean_reverting_assets_df.loc[mean_reverting_assets_df['date'] == previous_date, 'asset'].tolist()
    mean_reverting_assets_curr = mean_reverting_assets_df.loc[mean_reverting_assets_df['date'] == date, 'asset'].tolist()

    trendy_assets = list(set(trendy_assets_prev + trendy_assets_curr))
    mean_reverting_assets = list(set(mean_reverting_assets_prev + mean_reverting_assets_curr))

    filtered_assets = trendy_assets + mean_reverting_assets
    future_assets_returns = pd.DataFrame(index=[previous_date], columns=filtered_assets, dtype=float).fillna(0.0)

    # Trendy Assets: Cumulative Returns in the last period
    for asset in trendy_assets:
        #print(f" Calculate future returns for {asset} using {previous_dataframe_assets[asset]}")
        future_return = (1 + previous_dataframe_assets[asset]).prod() - 1
        future_assets_returns.loc[previous_date, asset] = float(future_return)
        #print(f"Future returns for {asset}: {future_assets_returns.loc[previous_date, asset]}")

    # Mean Reverting Assets: Mean of the returns in the last period
    for asset in mean_reverting_assets:
        #print(f" Calculate future returns for {asset} using {previous_dataframe_assets[asset]}")
        future_return = previous_dataframe_assets[asset].mean()
        future_assets_returns.loc[previous_date, asset] = float(future_return) 
        #print(f"Future returns for {asset}: {future_assets_returns.loc[previous_date, asset]}")

    return future_assets_returns
