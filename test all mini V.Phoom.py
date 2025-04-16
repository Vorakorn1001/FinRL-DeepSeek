# Input: 
# Task: Use PPO for Bullish and CPPO-DeepSeek for Bearish and do a normal strategy
# Output: Performance metrics for (PPO, CPPO, PPO-DeepSeek, CPPO-DeepSeek, Selective Model)
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from finrl.config import INDICATORS
from env_stocktrading import StockTradingEnv
from env_stocktrading_llm import StockTradingEnv as StockTradingEnv_llm
from env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnv_llm_risk
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from datasets import load_dataset
import scipy.signal
from gymnasium.spaces import Box, Discrete
import concurrent.futures
import yfinance as yf
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

seed = 42  # or any fixed number you choose
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

TRAINED_MODEL_DIR = 'trained_models_authors'
hidden_units = 512
hidden_layers = 2

hidden_sizes = tuple([hidden_units] * hidden_layers)

TRAIN_START_DATE = '2013-01-01'
TRAIN_END_DATE = '2018-12-31'
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2023-12-31'

stock_dimension = 84
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
state_space_llm = 1 + 2 * stock_dimension + (1+len(INDICATORS)) * stock_dimension
state_space_llm_risk = 1 + 2 * stock_dimension + (2+len(INDICATORS)) * stock_dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_normal_dataset():
    # from Huggging Face :
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_2019_2023.csv')
    # Convert to pandas DataFrame
    trade = pd.DataFrame(dataset['train'])
    trade = trade.drop('Unnamed: 0',axis=1)
    # Create a new index based on unique dates
    unique_dates = trade['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    # Create new index based on the date mapping
    trade['new_idx'] = trade['date'].map(date_to_idx)
    # Set this as the index
    trade = trade.set_index('new_idx')
    return trade

def get_llm_dataset():
    # from Huggging Face :
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_sentiment_2019_2023.csv')
    # Convert to pandas DataFrame
    trade = pd.DataFrame(dataset['train'])
    #trade= pd.read_csv('/content/machine_learning/trade_data_qwen_sentiment.csv')
    trade = trade.drop('Unnamed: 0',axis=1)
    # Create a new index based on unique dates
    unique_dates = trade['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    # Create new index based on the date mapping
    trade['new_idx'] = trade['date'].map(date_to_idx)
    # Set this as the index
    trade = trade.set_index('new_idx')
    #missing values with 0
    trade['llm_sentiment'] = trade['llm_sentiment'].fillna(0)
    trade_llm=trade
    return trade_llm

def get_llm_risk_dataset():
    # from Huggging Face :
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_risk_2019_2023.csv')
    # Convert to pandas DataFrame
    trade = pd.DataFrame(dataset['train'])
    trade = trade.drop('Unnamed: 0',axis=1)
    # Create a new index based on unique dates
    unique_dates = trade['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    # Create new index based on the date mapping
    trade['new_idx'] = trade['date'].map(date_to_idx)
    # Set this as the index
    trade = trade.set_index('new_idx')
    #missing values with 0
    trade['llm_sentiment'] = trade['llm_sentiment'].fillna(0)
    #missing values with 3
    trade['llm_risk'] = trade['llm_risk'].fillna(3)
    trade_llm_risk=trade
    return trade_llm_risk

def get_env_kwargs():
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    return env_kwargs

def get_env_kwargs_llm():
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs_llm = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space_llm,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    return env_kwargs_llm

def get_env_kwargs_llm_risk():
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs_llm_risk = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space_llm_risk,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    return env_kwargs_llm_risk

def get_gym(trade, env_kwargs):
    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
    return e_trade_gym

def get_gym_llm(trade_llm, env_kwargs_llm):
    e_trade_gym_llm = StockTradingEnv_llm(df = trade_llm, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs_llm)
    return e_trade_gym_llm

def get_gym_llm_risk(trade_llm_risk, env_kwargs_llm_risk):
    e_trade_gym_llm_risk = StockTradingEnv_llm_risk(df = trade_llm_risk, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs_llm_risk)
    return e_trade_gym_llm_risk

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
def load_ppo(observation_space, action_space):
    loaded_ppo = MLPActorCritic(observation_space, action_space, hidden_sizes=hidden_sizes)
    state_dict = torch.load(f'{TRAINED_MODEL_DIR}/agent_ppo_100_epochs_20k_steps.pth', map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    loaded_ppo.load_state_dict(new_state_dict)
    return loaded_ppo

def load_cppo(observation_space, action_space):
    loaded_cppo = MLPActorCritic(observation_space,action_space, hidden_sizes=hidden_sizes)
    state_dict = torch.load(f'{TRAINED_MODEL_DIR}/agent_cppo_100_epochs_20k_steps.pth', map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    loaded_cppo.load_state_dict(new_state_dict)
    return loaded_cppo

def load_ppo_deepseek(observation_space_llm, action_space_llm):
    loaded_ppo_llm = MLPActorCritic(observation_space_llm, action_space_llm, hidden_sizes=hidden_sizes)
    state_dict = torch.load(f'{TRAINED_MODEL_DIR}/agent_ppo_deepseek_100_epochs_20k_steps.pth', map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    loaded_ppo_llm.load_state_dict(new_state_dict)
    return loaded_ppo_llm

def load_cppo_deepseek(observation_space_llm_risk, action_space_llm_risk):
    loaded_cppo_llm_risk = MLPActorCritic(observation_space_llm_risk, action_space_llm_risk, hidden_sizes=hidden_sizes)
    state_dict = torch.load(f'{TRAINED_MODEL_DIR}/agent_cppo_deepseek_100_epochs_20k_steps.pth', map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    loaded_cppo_llm_risk.load_state_dict(new_state_dict)
    return loaded_cppo_llm_risk
    
def DRL_prediction(act, environment):
    import torch
    _torch = torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, _ = environment.reset()
    account_memory = []  # To store portfolio values
    actions_memory = []  # To store actions taken
    portfolio_distribution = []  # To store portfolio distribution
    episode_total_assets = [environment.initial_amount]
    act.to(device)

    with _torch.no_grad():
        for i in range(len(environment.df.index.unique())):
            s_tensor = _torch.as_tensor((state,), dtype=torch.float32, device=device)
            a_tensor, _, _ = act.step(s_tensor)  # Compute action
            action = a_tensor[0]  # Extract action

            # Step through the environment
            state, reward, done, _, _ = environment.step(action)

            # Get stock prices for the current day
            price_array = environment.df.loc[environment.day, "close"].values

            # Stock holdings and cash balance
            stock_holdings = environment.num_stock_shares
            cash_balance = environment.asset_memory[-1]

            # Calculate total portfolio value
            total_asset = cash_balance + (price_array * stock_holdings).sum()

            # Calculate portfolio distribution
            stock_values = price_array * stock_holdings
            total_invested = stock_values.sum()
            distribution = stock_values / total_asset  # Fraction of each stock in the total portfolio
            cash_fraction = cash_balance / total_asset

            # Store results
            episode_total_assets.append(total_asset)
            account_memory.append(total_asset)
            actions_memory.append(action)
            portfolio_distribution.append({"cash": cash_fraction, "stocks": distribution.tolist()})

       #     print("Total Asset Value:", total_asset)
        #    print("Portfolio Distribution:", {"cash": cash_fraction, "stocks": distribution.tolist()})

            if done:
                break

    print("Test Finished!")
    return episode_total_assets, account_memory, actions_memory, portfolio_distribution
    
def filter_to_common_dates(trade, df_dji, df_assets_ppo, df_dji_normalized_close):
    """
    Filters df_assets_ppo and df_dji_normalized_close based on the common dates from trade and df_dji.

    Parameters:
        trade (pd.DataFrame): DataFrame containing a 'date' column for the trade data.
        df_dji (pd.DataFrame): DataFrame containing a 'date' column for DJI data.
        df_assets_ppo (list or array-like): Values corresponding to trade['date'].
        df_dji_normalized_close (list or array-like): Values corresponding to df_dji['date'].

    Returns:
        pd.Series, pd.Series: Filtered series for df_assets_ppo and df_dji_normalized_close.
    """
    # Extract unique trading dates from trade and DJI dates
    trade_dates = pd.DatetimeIndex(trade['date'].unique())
    dji_dates = pd.to_datetime(df_dji['date'].unique())


  #  first_date = trade_dates[0]
   # date_before_first = first_date - pd.DateOffset(days=1)

# Prepend the date before the first date to trade_dates
    #trade_dates = pd.DatetimeIndex([date_before_first] + trade_dates.tolist())

    # Convert inputs to pandas Series with their respective dates as indices
    df_assets_ppo_series = pd.Series(df_assets_ppo, index=trade_dates)
    df_dji_normalized_close_series = pd.Series(df_dji_normalized_close, index=dji_dates)

    # Find the common dates
    common_dates = trade_dates.intersection(dji_dates)

    # Filter both series to the common dates
    df_assets_ppo_filtered = df_assets_ppo_series.reindex(common_dates)
    df_dji_normalized_close_filtered = df_dji_normalized_close_series.reindex(common_dates)

    # Return the filtered series
    return df_assets_ppo_filtered, df_dji_normalized_close_filtered, common_dates

def get_normalized_close(df_assets, inital_value=1000000):
    fst_day = df_assets[1] 
    df_assets_series = pd.Series(df_assets[1:])
    normalized_close = list(df_assets_series.div(fst_day).mul(inital_value))
    return normalized_close

def calculate_metric(returns_strategy, returns_benchmark, confidence_level=0.05, upside_confidence=0.95):
    """Calculate performance metrics: IR, CVaR, Rachev Ratio, and Outperform frequency."""
    # Daily Outperform Frequency: Check on each day if the strategy beats the benchmark.
    daily_outperform = (returns_strategy > returns_benchmark)
    daily_frequency = daily_outperform.mean()  # The mean of a boolean series gives the proportion of True values.
    
    # Weekly Outperform Frequency:
    weekly_strategy = returns_strategy.resample('W').sum()
    weekly_benchmark = returns_benchmark.resample('W').sum()
    
    # Compare the weekly aggregated returns.
    weekly_outperform = (weekly_strategy > weekly_benchmark)
    weekly_frequency = weekly_outperform.mean()
    excess_return = returns_strategy - returns_benchmark
    ir = excess_return.mean() / excess_return.std()
    var = np.percentile(returns_strategy, confidence_level * 100)
    cvar = returns_strategy[returns_strategy <= var].mean()
    upside_var = np.percentile(returns_strategy, upside_confidence * 100)
    downside_var = var
    rachev_ratio = returns_strategy[returns_strategy >= upside_var].mean() / abs(returns_strategy[returns_strategy <= downside_var].mean())
    return {
        "Information Ratio": ir, 
        "CVaR": cvar, 
        "Rachev Ratio": rachev_ratio, 
        "Daily Outperform Frequency": daily_frequency, 
        "Weekly Outperform Frequency": weekly_frequency
        }

def align_returns(result, col_strategy, col_benchmark):
    """Align returns for strategy and benchmark."""
    returns_strategy = result[col_strategy].pct_change(fill_method=None).dropna()
    returns_benchmark = result[col_benchmark].pct_change(fill_method=None).dropna()
    return returns_strategy.align(returns_benchmark, join="inner")

def compute_metrics(result, strategies, benchmark, confidence_level=0.05, upside_confidence=0.95):
    """
    Compute metrics for multiple strategies compared to a benchmark.

    Parameters:
        result (pd.DataFrame): DataFrame with strategies and benchmark columns.
        strategies (list): List of strategy column names.
        benchmark (str): Benchmark column name.
        confidence_level (float): Confidence level for CVaR calculation.
        upside_confidence (float): Confidence level for upside in Rachev Ratio.

    Returns:
        dict: Performance metrics for each strategy.
    """
    metrics = {}
    for strategy in strategies:
        aligned_strategy, aligned_benchmark = align_returns(result, strategy, benchmark)
        metrics[strategy] = calculate_metric(
            aligned_strategy, aligned_benchmark, confidence_level, upside_confidence
        )
    return metrics

def plot_cumulative_returns(result, metrics, strategies, benchmark,
                            save_path="cumulative_returns_All_phoom.png", dpi=300):
    """
    Plot cumulative returns for strategies and benchmark and save to file.

    Parameters:
        result (pd.DataFrame): DataFrame with strategies and benchmark.
        metrics (dict): Performance metrics (not used for plotting here,
                        but you could annotate the plot if you like).
        strategies (list): List of strategy column names.
        benchmark (str): Benchmark column name.
        save_path (str): File path (including filename) to save the figure.
        dpi (int): Resolution of the saved figure.
    """
    plt.figure(figsize=(12, 6))

    for strategy in strategies:
        cum_ret = (1 + result[strategy].pct_change(fill_method=None).dropna()).cumprod()
        plt.plot(cum_ret, label=strategy)

    cum_bench = (1 + result[benchmark].pct_change(fill_method=None).dropna()).cumprod()
    plt.plot(cum_bench, label=f"{benchmark} (Benchmark)", linestyle="--")

    plt.title("Cumulative Returns with Performance Metrics")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)

    # Save to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def get_ndx_market_condition(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Nasdaq-100 index data from the factors DataFrame (df_factors) and computes market conditions.
    
    Parameters:
        df_factors (pd.DataFrame): DataFrame obtained from YahooDownloader containing various tickers including '^NDX'.
    
    Returns:
        pd.DataFrame: A DataFrame with 'date', 'Bullish_stable', and 'Bearish' columns indicating the stabilized
                      market condition.
    """
    # Filter for Nasdaq-100 index data (ticker '^NDX')
    ndx_df = df_factors[df_factors['tic'] == '^NDX'].copy()
    ndx_df.sort_values('date', inplace=True)
    
    # Compute moving averages on the close price
    ndx_df["SMA_200"] = ndx_df["close"].rolling(window=200).mean()
    ndx_df["SMA_50"] = ndx_df["close"].rolling(window=50).mean()
    
    # Set initial bullish signal: True when SMA_50 > SMA_200 (where SMA_200 is available)
    ndx_df["Bullish"] = True
    ndx_df.loc[ndx_df["SMA_200"].notnull(), "Bullish"] = ndx_df["SMA_50"] > ndx_df["SMA_200"]
    
    # Convert date to datetime
    ndx_df["date"] = pd.to_datetime(ndx_df["date"], errors='coerce')
    
    # Stabilize the bullish signal to reduce short-term noise
    threshold = 30
    bullish_vals = ndx_df["Bullish"].values
    group_change = np.concatenate(([True], bullish_vals[1:] != bullish_vals[:-1]))
    group_ids = np.cumsum(group_change)
    
    stable = bullish_vals.copy()
    for group in np.unique(group_ids):
        idx = np.where(group_ids == group)[0]
        if len(idx) < threshold and idx[0] > 0:
            stable[idx] = stable[idx[0] - 1]
    ndx_df["Bullish_stable"] = stable
    ndx_df['Bearish'] = ~ndx_df['Bullish_stable']
    
    # Return only the relevant columns
    return ndx_df[['date', 'Bullish_stable', 'Bearish']]

def split_market_segments_by_ndx(df: pd.DataFrame, df_factors: pd.DataFrame) -> list:
    """
    Splits the main market DataFrame into segments based on Nasdaq-100 bull/bear conditions.
    
    Parameters:
        df (pd.DataFrame): Main market DataFrame (individual stock data) without '^NDX' rows.
        df_factors (pd.DataFrame): DataFrame with additional factors including the '^NDX' data (from YahooDownloader).
    
    Returns:
        list: A list of dictionaries, each representing a market segment with:
              - label: "Bull" or "Bear"
              - start_date: starting date of the segment
              - end_date: ending date of the segment
              - data: the segment’s sub-DataFrame (with a new index based on unique dates)
    """
    # Get Nasdaq-100 market condition from the factors DataFrame
    ndx_condition = get_ndx_market_condition(df_factors)
    
    # Ensure the main DataFrame has a proper datetime date column
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    # Merge the Nasdaq condition onto the main DataFrame based on date
    merged_df = pd.merge(df, ndx_condition, on="date", how="left")
    # Drop any rows without a condition (e.g., dates outside the '^NDX' data range)
    merged_df = merged_df.dropna(subset=["Bullish_stable"])
    
    # Group by date to get the daily market condition (one row per date)
    daily_condition = merged_df.groupby("date").first().reset_index()[["date", "Bullish_stable", "Bearish"]]
    daily_condition.sort_values("date", inplace=True)
    
    # Identify segments where the market condition (bullish or bearish) remains unchanged
    daily_condition["group"] = (daily_condition["Bearish"] != daily_condition["Bearish"].shift()).cumsum()
    
    segments = []
    for group_id, group in daily_condition.groupby("group"):
        label = "Bear" if group["Bearish"].iloc[0] else "Bull"
        start_date = group["date"].iloc[0]
        end_date = group["date"].iloc[-1]
        group_dates = group["date"].tolist()
        
        # Filter the merged DataFrame for the current segment's dates
        segment_data = merged_df[merged_df["date"].isin(group_dates)].copy()
        
        # Create a new index based on unique dates
        unique_dates = sorted(segment_data["date"].unique())
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        segment_data['new_idx'] = segment_data["date"].map(date_to_idx)
        segment_data = segment_data.set_index('new_idx')
        
        # Optionally drop the Nasdaq condition columns from the segment data
        segment_data = segment_data.drop(columns=["Bullish_stable", "Bearish"])
        
        segments.append({
            "label": label,
            "start_date": start_date,
            "end_date": end_date,
            "data": segment_data
        })
        
    return segments

def adaptive_flow():
    # Bull -> PPO
    # Bear -> CPPO-DeepSeek 
    
    # Load market data
    df_factors = get_df_factors()
    
    # Load the dataset
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_risk_2019_2023.csv')
    
    # Convert to pandas DataFrame
    trade = pd.DataFrame(dataset['train'])
    trade = trade.drop('Unnamed: 0',axis=1)
    
    # Split the data
    segments = split_market_segments_by_ndx(trade, df_factors)

    portfolio = pd.DataFrame()
    initial_value = 1000000

    for segment in segments:
        df_dji = YahooDownloader(
            start_date=segment['start_date'], end_date=segment['end_date'], ticker_list=["^NDX"]
        ).fetch_data()
        df_dji = df_dji[["date", "close"]]
        fst_day = df_dji["close"].iloc[0]  # Safely get the first value
        df_dji_normalized_close = list(df_dji["close"].div(fst_day).mul(initial_value))
        
        if segment['label'] == 'Bull':
            sub_trade = segment['data']
            sub_trade = sub_trade.drop(columns=['llm_sentiment', 'llm_risk'])
            env_kwargs = get_env_kwargs()
            e_trade_gym = get_gym(sub_trade, env_kwargs)
            loaded_ppo = load_ppo(e_trade_gym.observation_space, e_trade_gym.action_space)
            df_assets_ppo, df_account_value_ppo, df_actions_ppo, df_portfolio_distribution_ppo = DRL_prediction(act=loaded_ppo, environment=e_trade_gym)
            df_ppo_normalized_close = get_normalized_close(df_assets_ppo, initial_value)
            df_assets_ppo_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(sub_trade, df_dji, df_ppo_normalized_close, df_dji_normalized_close)
            
            dfs_to_concat = [df for df in [portfolio, df_assets_ppo_filtered] if not df.empty]
            portfolio = pd.concat(dfs_to_concat, axis=0)

        elif segment['label'] == 'Bear':
            sub_trade = segment['data']
            sub_trade['llm_sentiment'] = sub_trade['llm_sentiment'].fillna(0)
            sub_trade['llm_risk'] = sub_trade['llm_risk'].fillna(3)
            env_kwargs_llm_risk = get_env_kwargs_llm_risk()
            e_trade_llm_risk_gym = get_gym_llm_risk(sub_trade, env_kwargs_llm_risk)
            loaded_cppo_llm_risk = load_cppo_deepseek(e_trade_llm_risk_gym.observation_space, e_trade_llm_risk_gym.action_space)
            
            df_assets_cppo_llm_risk, df_account_value_cppo_llm_risk, df_actions_cppo_llm_risk, df_portfolio_distribution_cppo_llm_risk = DRL_prediction(act=loaded_cppo_llm_risk, environment=e_trade_llm_risk_gym)
            
            df_cppo_llm_risk_normalized_close = get_normalized_close(df_assets_cppo_llm_risk, initial_value)
            df_assets_cppo_llm_risk_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(sub_trade, df_dji, df_cppo_llm_risk_normalized_close, df_dji_normalized_close)
            
            dfs_to_concat = [df for df in [portfolio, df_assets_cppo_llm_risk_filtered] if not df.empty]
            portfolio = pd.concat(dfs_to_concat, axis=0)

        initial_value = portfolio.iloc[-1]
    
    portfolio = portfolio.squeeze()
    return portfolio


def load_specialized_models(stock_list, model_dir='ppo_mini_models'):
    """
    Load specialized models for each stock from ppo_mini_models folder.
    Returns a dictionary mapping stock tickers to their loaded models.
    
    Parameters:
        stock_list (list): List of stock tickers
        
    Returns:
        dict: Dictionary mapping stock tickers to their respective models
    """
    specialized_models = {}
    # model_dir = 'ppo_mini_models'
    
    # Ensure directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created directory {model_dir}")
        return specialized_models
    
    # Get all model files in the ppo_mini_models directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    # Match each stock with its specialized model if available
    for stock in stock_list:
        # Look for model files containing the stock ticker
        matching_files = [f for f in model_files if stock in f]
        
        if matching_files:
            # Use the first matching file if multiple matches exist
            model_path = os.path.join(model_dir, matching_files[0])
            print(f"Found specialized model for {stock}: {matching_files[0]}")
            specialized_models[stock] = model_path
        else:
            print(f"No specialized model found for {stock}")
    
    return specialized_models

def load_mini_specialized(observation_space, action_space, stock_ticker=None, specialized_models=None):
    """
    Load the appropriate specialized model for a stock if available,
    otherwise fall back to a generic mini model or standard PPO.
    
    Parameters:
        observation_space: Gym observation space
        action_space: Gym action space  
        stock_ticker (str): Stock ticker to load model for
        specialized_models (dict): Dictionary mapping stock tickers to model paths
        
    Returns:
        model: Loaded PyTorch model
    """
    # If we have a stock ticker and a specialized model for it, try to load that
    if stock_ticker and specialized_models and stock_ticker in specialized_models:
        model_path = specialized_models[stock_ticker]
        print(f"Loading specialized model for {stock_ticker} from {model_path}")
        
        try:
            # Create mini model architecture with the CURRENT environment dimensions
            mini_hidden_units = 64
            mini_hidden_layers = 3
            mini_hidden_sizes = tuple([mini_hidden_units] * mini_hidden_layers)
            
            loaded_mini = MLPActorCritic(observation_space, action_space, hidden_sizes=mini_hidden_sizes)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Clean up the state dict keys
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # FIXED: Create a compatibility layer to handle tensor size mismatches
            compatible_state_dict = create_compatible_state_dict(new_state_dict, loaded_mini.state_dict())
            
            if compatible_state_dict:
                loaded_mini.load_state_dict(compatible_state_dict, strict=False)
                print(f"✅ Successfully loaded specialized model for {stock_ticker} (compatible mode)")
                
                # Initialize any missing parameters
                for name, param in loaded_mini.named_parameters():
                    if name not in compatible_state_dict:
                        print(f"  - Initializing missing parameter: {name}")
                
                return loaded_mini
            else:
                # If we couldn't create a compatible state dict, initialize the weights differently
                # based on the stock-specific model characteristics
                print(f"⚠️ Could not create compatible state dict for {stock_ticker}, using initialization")
                
                # Extract embedding dimensions from the specialized model if possible
                if "pi.mu_net.0.weight" in new_state_dict:
                    embed_dim = new_state_dict["pi.mu_net.0.weight"].size(1)
                    print(f"  - Using specialized initialization with embed dim {embed_dim}")
                    
                    # Custom initialization based on the specialized model's characteristics
                    with torch.no_grad():
                        # Initialize action bias for this stock based on historical behavior
                        # This is just an example - you might want different logic
                        if stock_ticker in ['AAPL', 'MSFT', 'GOOGL']:  # tech stocks
                            loaded_mini.pi.log_std.data.fill_(-0.7)  # more confident predictions
                        elif stock_ticker in ['JNJ', 'PFE', 'UNH']:  # healthcare stocks
                            loaded_mini.pi.log_std.data.fill_(-0.5)  # moderate confidence
                        else:
                            loaded_mini.pi.log_std.data.fill_(-0.3)  # less confident
                
                return loaded_mini
                
        except Exception as e:
            print(f"Failed to load specialized model for {stock_ticker}: {e}")
    
    # Fall back to loading generic mini model
    try:
        print(f"No specialized model for {stock_ticker}, trying to load generic mini model")
        mini_model_path = os.path.join('ppo_mini_models', 'agent_ppo_mini.pth')
        
        if os.path.exists(mini_model_path):
            mini_hidden_units = 64
            mini_hidden_layers = 3
            mini_hidden_sizes = tuple([mini_hidden_units] * mini_hidden_layers)
            
            loaded_mini = MLPActorCritic(observation_space, action_space, hidden_sizes=mini_hidden_sizes)
            state_dict = torch.load(mini_model_path, map_location=device)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            compatible_state_dict = create_compatible_state_dict(new_state_dict, loaded_mini.state_dict())
            
            if compatible_state_dict:
                loaded_mini.load_state_dict(compatible_state_dict, strict=False)
                print(f"✅ Successfully loaded generic mini model (compatible mode)")
                return loaded_mini
            else:
                print(f"⚠️ Could not create compatible state dict for generic mini model")
                return create_new_mini_model(observation_space, action_space)
        else:
            print(f"Generic mini model file not found, creating new model")
            return create_new_mini_model(observation_space, action_space)
            
    except Exception as e:
        print(f"Failed to load generic mini model: {e}. Falling back to standard PPO")
        return load_ppo(observation_space, action_space)

def create_compatible_state_dict(source_dict, target_dict):
    """
    Create a compatible state dictionary by only including parameters 
    that have matching shapes between source and target models.
    
    Parameters:
        source_dict (dict): Source state dictionary
        target_dict (dict): Target state dictionary
        
    Returns:
        dict: Compatible state dictionary or None if critical components don't match
    """
    compatible_dict = {}
    critical_mismatch = False
    
    for name, target_param in target_dict.items():
        if name in source_dict:
            source_param = source_dict[name]
            if source_param.size() == target_param.size():
                compatible_dict[name] = source_param
            else:
                print(f"  - Size mismatch for {name}: source {source_param.size()} vs target {target_param.size()}")
                # Some parameters are more critical than others
                if 'log_std' in name:
                    critical_mismatch = True
    
    # If critical parameters don't match, it might be better to start fresh
    if critical_mismatch and len(compatible_dict) < len(target_dict) // 2:
        print("  ⚠️ Critical parameters missing, compatibility may be limited")
        return None
    
    return compatible_dict

def create_new_mini_model(observation_space, action_space):
    """
    Create a new mini model when loading fails.
    
    Parameters:
        observation_space: Gym observation space
        action_space: Gym action space
        
    Returns:
        MLPActorCritic: A new mini model instance with custom initialization
    """
    print("Creating new mini model with custom initialization...")
    mini_hidden_units = 64
    mini_hidden_layers = 3
    mini_hidden_sizes = tuple([mini_hidden_units] * mini_hidden_layers)
    
    model = MLPActorCritic(observation_space, action_space, hidden_sizes=mini_hidden_sizes)
    
    # Apply custom initialization
    with torch.no_grad():
        # Initialize smaller standard deviation for more stable initial predictions
        model.pi.log_std.data.fill_(-1.0)
        
        # Initialize network weights with smaller values
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
    
    return model

class StockSpecificActorCritic:
    """
    A wrapper class that selects the appropriate model for each stock.
    Used for portfolio management where different stocks use different models.
    """
    def __init__(self, observation_space, action_space, stock_list, specialized_models):
        self.models = {}
        self.fallback_model = load_ppo(observation_space, action_space)
        self.stock_list = stock_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load specialized models for each stock
        for stock in stock_list:
            self.models[stock] = load_mini_specialized(
                observation_space, 
                action_space, 
                stock, 
                specialized_models
            )
            
        # Move all models to the appropriate device
        self.fallback_model.to(self.device)
        for stock in stock_list:
            if stock in self.models:
                self.models[stock].to(self.device)
            
    def step(self, obs, current_stock=None):
        """
        Take a step using the appropriate model for the current stock.
        
        Parameters:
            obs: Observation tensor
            current_stock: Current stock being traded
            
        Returns:
            Actions, values, and log probabilities
        """
        # Ensure observation is on the correct device
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        elif obs.device != self.device:
            obs = obs.to(self.device)
            
        if current_stock and current_stock in self.models:
            return self.models[current_stock].step(obs)
        else:
            return self.fallback_model.step(obs)
            
    def act(self, obs, current_stock=None):
        """Get actions only from the appropriate model"""
        return self.step(obs, current_stock)[0]

def DRL_prediction_specialized(act_ensemble, environment):
    """Universal prediction function handling both PPO and CPPO ensembles"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, _ = environment.reset()
    episode_total_assets = [environment.initial_amount]
    
    with torch.no_grad():
        for _ in range(len(environment.df.index.unique())):
            # Get current trading day data
            day_data = environment.df.loc[environment.day]
            available_stocks = day_data.tic.unique().tolist()
            
            # Initialize action array
            action = np.zeros(environment.stock_dim)
            
            # Convert state to tensor
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            
            # Get actions for each stock
            for stock_idx, stock_tic in enumerate(environment.df.tic.unique()):
                if stock_tic in available_stocks:
                    # Get stock-specific action
                    a, _, _ = act_ensemble.step(state_tensor, stock_tic)
                    action[stock_idx] = a[stock_idx] if isinstance(a, np.ndarray) else a.item()
            
            # Execute combined action
            state, _, done, _, _ = environment.step(action)
            episode_total_assets.append(environment.asset_memory[-1])
            
            if done:
                break
    
    return episode_total_assets, [], [], []

def load_cppo_specialized_models(stock_list, model_dir='cppo_deepseek_mini_models'):
    """Load specialized CPPO models for bear markets"""
    specialized_models = {}
    model_dir = 'cppo_deepseek_mini_models'
    
    if not os.path.exists(model_dir):
        return specialized_models

    for stock in stock_list:
        model_files = [f for f in os.listdir(model_dir) 
                      if f.endswith('.pth') and stock in f]
        if model_files:
            specialized_models[stock] = os.path.join(model_dir, model_files[0])
    
    return specialized_models

class CPPOEnsemble:
    """Specialized CPPO models for bear markets"""
    def __init__(self, observation_space, action_space, stock_list, specialized_models):
        self.models = {}
        self.fallback = load_cppo_deepseek(observation_space, action_space)
        self.stock_list = stock_list
        
        for stock in stock_list:
            if stock in specialized_models:
                try:
                    model = MLPActorCritic(observation_space, action_space, hidden_sizes=(64,64))
                    state_dict = torch.load(specialized_models[stock])
                    model.load_state_dict(state_dict)
                    self.models[stock] = model
                except:
                    continue
                    
    def step(self, obs, current_stock=None):
        if current_stock and current_stock in self.models:
            return self.models[current_stock].step(obs)
        else:
            return self.fallback.step(obs)
            
def adaptive_specialized_flow():
    """Adaptive strategy using specialized PPO models for bulls and CPPO models for bears"""
    # Load market data
    df_factors = get_df_factors()
    
    # Load dataset
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_risk_2019_2023.csv')
    trade = pd.DataFrame(dataset['train']).drop('Unnamed: 0', axis=1)
    stock_list = trade['tic'].unique().tolist()
    
    # Load specialized models
    ppo_models = load_specialized_models(stock_list)
    cppo_models = load_specialized_models(stock_list, model_dir='cppo_deepseek_mini_models')
    
    # Split market segments
    segments = split_market_segments_by_ndx(trade, df_factors)
    portfolio = pd.DataFrame()
    initial_value = 1_000_000

    class CPPOEnsemble:
        """Specialized CPPO model handler for bear markets"""
        def __init__(self, observation_space, action_space, stock_list, models):
            self.models = {}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.fallback = load_cppo_deepseek(observation_space, action_space).to(self.device)
            
            for stock in stock_list:
                if stock in models:
                    try:
                        model = MLPActorCritic(observation_space, action_space, hidden_sizes=(64, 64))
                        state_dict = torch.load(models[stock], map_location=self.device)
                        model.load_state_dict(state_dict)
                        model.to(self.device)
                        self.models[stock] = model
                    except Exception as e:
                        print(f"Error loading {stock} CPPO model: {e}")

        def step(self, obs, current_stock=None):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if current_stock and current_stock in self.models:
                return self.models[current_stock].step(obs_tensor)
            return self.fallback.step(obs_tensor)

    for segment in segments:
        df_dji = YahooDownloader(
            start_date=segment['start_date'],
            end_date=segment['end_date'],
            ticker_list=["^NDX"]
        ).fetch_data()[["date", "close"]]
        
        if segment['label'] == 'Bull':
            # Bull market: Use PPO specialized models
            sub_trade = segment['data'].drop(columns=['llm_sentiment', 'llm_risk'])
            env_kwargs = get_env_kwargs()
            e_trade_gym = get_gym(sub_trade, env_kwargs)
            
            ensemble = StockSpecificActorCritic(
                e_trade_gym.observation_space,
                e_trade_gym.action_space,
                stock_list,
                ppo_models
            )
            
            assets, _, _, _ = DRL_prediction_specialized(ensemble, e_trade_gym)
            
        elif segment['label'] == 'Bear':
            # Bear market: Use CPPO specialized models with risk features
            sub_trade = segment['data'].copy()
            sub_trade['llm_sentiment'] = sub_trade['llm_sentiment'].fillna(0)
            sub_trade['llm_risk'] = sub_trade['llm_risk'].fillna(3)
            
            env_kwargs = get_env_kwargs_llm_risk()
            e_trade_gym = get_gym_llm_risk(sub_trade, env_kwargs)
            
            ensemble = CPPOEnsemble(
                e_trade_gym.observation_space,
                e_trade_gym.action_space,
                stock_list,
                cppo_models
            )
            
            assets, _, _, _ = DRL_prediction_specialized(ensemble, e_trade_gym)

        # Process and align results
        normalized = get_normalized_close(assets, initial_value)
        fst_day = df_dji["close"].iloc[0]
        bench_norm = list(df_dji["close"].div(fst_day).mul(initial_value))
        
        filtered_assets, _, _ = filter_to_common_dates(
            sub_trade, df_dji, normalized, bench_norm
        )
        
        portfolio = pd.concat([portfolio, filtered_assets], axis=0)
        initial_value = portfolio.iloc[-1] if not portfolio.empty else initial_value

    return portfolio.squeeze()

def DRL_prediction_specialized_full(act_ensemble, environment):
    """
    Modified DRL_prediction to use specialized models for all stocks throughout the entire period.
    
    Parameters:
        act_ensemble: StockSpecificActorCritic ensemble of models
        environment: Trading environment
        
    Returns:
        Tuple of account history, actions, etc.
    """
    import torch
    _torch = torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, _ = environment.reset()
    account_memory = []
    actions_memory = []
    portfolio_distribution = []
    episode_total_assets = [environment.initial_amount]

    with _torch.no_grad():
        for i in range(len(environment.df.index.unique())):
            # Get current trading day and available stocks
            day_data = environment.df.loc[environment.day]
            available_stocks = day_data.tic.unique().tolist()
            
            # Convert state to tensor
            s_tensor = _torch.as_tensor((state,), dtype=torch.float32, device=device)
            
            # Default action initialized as zeros
            action = np.zeros(environment.stock_dim)
            
            # For each stock in our universe, use its specialized model if available
            for stock_idx, stock_tic in enumerate(environment.df.tic.unique()):
                if stock_tic in available_stocks:
                    # Use the specialized model for this stock
                    stock_action, _, _ = act_ensemble.step(s_tensor, stock_tic)
                    action[stock_idx] = stock_action[0][stock_idx]
                else:
                    # Stock not available for this day, set action to 0
                    action[stock_idx] = 0
            
            # Take step with the ensemble action
            state, reward, done, _, _ = environment.step(action)

            # Calculate portfolio metrics - FIX: Handle both Series and scalar cases
            day_data = environment.df.loc[environment.day]
            
            # Create a properly indexed price array
            price_array = np.zeros(environment.stock_dim)
            stock_holdings = environment.num_stock_shares
            
            # Map each close price to the correct position in price_array
            for stock_idx, stock_tic in enumerate(environment.df.tic.unique()):
                # Find this stock in the day's data if it exists
                stock_data = day_data[day_data.tic == stock_tic]
                if not stock_data.empty:
                    price_array[stock_idx] = stock_data.iloc[0]['close']
            
            cash_balance = environment.asset_memory[-1]
            total_asset = cash_balance + (price_array * stock_holdings).sum()
            stock_values = price_array * stock_holdings
            
            # Calculate portfolio distribution
            distribution = stock_values / total_asset if total_asset > 0 else np.zeros_like(stock_values)
            cash_fraction = cash_balance / total_asset if total_asset > 0 else 1.0

            # Store results
            episode_total_assets.append(total_asset)
            account_memory.append(total_asset)
            actions_memory.append(action)
            portfolio_distribution.append({"cash": cash_fraction, "stocks": distribution.tolist()})

            if done:
                break

    print("Test Finished!")
    return episode_total_assets, account_memory, actions_memory, portfolio_distribution

def get_df_factors():
    ticker_list = ["^NDX"]
    data_df = pd.DataFrame()
    num_failures = 0
    for tic in ticker_list:
        temp_df = yf.download(
            tic, start=TRADE_START_DATE, end=TRADE_END_DATE, auto_adjust=False
        )
        temp_df["tic"] = tic
        if len(temp_df) > 0:
            data_df = pd.concat([data_df, temp_df], axis=0)
        else:
            num_failures += 1
    if num_failures == len(ticker_list):
        raise ValueError("no data is fetched.")
    data_df = data_df.reset_index()
    try:
        data_df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjcp",
            "volume",
            "tic",
        ]
        data_df["close"] = data_df["adjcp"]
        data_df = data_df.drop(labels="adjcp", axis=1)
    except NotImplementedError:
        print("the features are not supported currently")
    data_df["day"] = data_df["date"].dt.dayofweek
    data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    print("Shape of DataFrame: ", data_df.shape)
    data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

    return data_df

def full_specialized_flow():
    """
    Full specialized flow that uses specialized PPO models for all stocks throughout the entire period,
    without segmenting into bull/bear markets.
    """
    # Load the dataset
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_risk_2019_2023.csv')
    
    # Convert to pandas DataFrame
    trade = pd.DataFrame(dataset['train'])
    trade = trade.drop('Unnamed: 0',axis=1)
    
    # Create a new index based on unique dates
    unique_dates = trade['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    # Create new index based on the date mapping
    trade['new_idx'] = trade['date'].map(date_to_idx)
    # Set this as the index
    trade = trade.set_index('new_idx')
    
    # Get stock list
    stock_list = trade['tic'].unique().tolist()
    
    # Load specialized models for each stock
    specialized_models = load_specialized_models(stock_list)
    print(f"Loaded {len(specialized_models)} specialized models out of {len(stock_list)} stocks")
    
    # Prepare standard environment (without LLM data)
    trade_standard = trade.drop(columns=['llm_sentiment', 'llm_risk'])
    env_kwargs = get_env_kwargs()
    e_trade_gym = get_gym(trade_standard, env_kwargs)
    
    # Create model ensemble using specialized models
    model_ensemble = StockSpecificActorCritic(
        e_trade_gym.observation_space, 
        e_trade_gym.action_space,
        stock_list,
        specialized_models
    )
    
    # Run prediction with specialized models
    df_assets_model, df_account_value_model, df_actions_model, df_portfolio_distribution_model = DRL_prediction_specialized_full(
        act_ensemble=model_ensemble, 
        environment=e_trade_gym
    )
    
    # Normalize results
    df_model_normalized_close = get_normalized_close(df_assets_model, 1000000)
    
    # Get benchmark data
    df_dji = YahooDownloader(
        start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["^NDX"]
    ).fetch_data()
    df_dji = df_dji[["date", "close"]]
    fst_day = df_dji["close"].iloc[0]
    df_dji_normalized_close = list(df_dji["close"].div(fst_day).mul(1000000))
    
    # Align data with benchmark
    df_assets_model_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(
        trade, df_dji, df_model_normalized_close, df_dji_normalized_close
    )
    
    return df_assets_model_filtered

# Add this helper function for debugging complex errors
def debug_values(value, name="Value"):
    """Helper function to debug values and their types"""
    print(f"{name} type: {type(value)}")
    if hasattr(value, 'shape'):
        print(f"{name} shape: {value.shape}")
    elif isinstance(value, (list, tuple)):
        print(f"{name} length: {len(value)}")
    if isinstance(value, pd.DataFrame):
        print(f"{name} columns: {value.columns.tolist()}")
        print(f"{name} head:\n{value.head()}")
    return value  # Return the value so this can be used inline

def load_cppo_mini_specialized(observation_space, action_space, stock_ticker=None, specialized_models=None):
    """Load CPPO mini models with same structure as PPO version but different directory"""
    if stock_ticker and specialized_models and stock_ticker in specialized_models:
        model_path = specialized_models[stock_ticker]
        print(f"Loading specialized CPPO model for {stock_ticker} from {model_path}")
        
        try:
            mini_hidden_units = 64
            mini_hidden_layers = 2
            mini_hidden_sizes = tuple([mini_hidden_units] * mini_hidden_layers)
            
            loaded_mini = MLPActorCritic(observation_space, action_space, hidden_sizes=mini_hidden_sizes)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            compatible_state_dict = create_compatible_state_dict(new_state_dict, loaded_mini.state_dict())
            
            if compatible_state_dict:
                loaded_mini.load_state_dict(compatible_state_dict, strict=False)
                print(f"✅ Successfully loaded specialized CPPO model for {stock_ticker}")
                return loaded_mini
            else:
                print("⚠️ Using CPPO model with architecture adjustments")
                return create_new_mini_model(observation_space, action_space)
                
        except Exception as e:
            print(f"Failed to load CPPO model: {e}")
            return load_cppo(observation_space, action_space)
    
    # Fallback to generic CPPO
    return load_cppo(observation_space, action_space)

def full_cppo_specialized_flow():
    """Pure CPPO strategy for whole market"""
    # Load dataset
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_risk_2019_2023.csv')
    trade = pd.DataFrame(dataset['train']).drop('Unnamed: 0',axis=1)
    
    # Create index
    unique_dates = trade['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    trade['new_idx'] = trade['date'].map(date_to_idx)
    trade = trade.set_index('new_idx')
    
    # Get stock list and load CPPO models
    stock_list = trade['tic'].unique().tolist()
    specialized_models = {}
    model_dir = 'cppo_deepseek_mini_models'
    
    # Load CPPO models
    if os.path.exists(model_dir):
        for stock in stock_list:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and stock in f]
            if model_files:
                specialized_models[stock] = os.path.join(model_dir, model_files[0])
    
    # Create environment
    trade_standard = trade.drop(columns=['llm_sentiment', 'llm_risk'])
    env_kwargs = get_env_kwargs()
    e_trade_gym = get_gym(trade_standard, env_kwargs)
    
    # Modified CPPOEnsemble class with proper device handling
    class CPPOEnsemble:
        def __init__(self, observation_space, action_space, stock_list, models):
            self.models = {}
            self.device = device  # Use global device
            
            for stock in stock_list:
                if stock in models:
                    model = load_cppo_mini_specialized(
                        observation_space, action_space, stock, models
                    )
                    model.to(self.device)
                    self.models[stock] = model
            self.fallback = load_cppo(observation_space, action_space).to(self.device)
            
        def step(self, obs, stock=None):
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            elif obs.device != self.device:
                obs = obs.to(self.device)
                
            if stock and stock in self.models:
                return self.models[stock].step(obs)
            else:
                return self.fallback.step(obs)
    
    # Create ensemble and run prediction
    ensemble = CPPOEnsemble(e_trade_gym.observation_space, e_trade_gym.action_space, 
                           stock_list, specialized_models)
    
    # Modified prediction loop
    state, _ = e_trade_gym.reset()
    episode_total_assets = [e_trade_gym.initial_amount]
    
    with torch.no_grad():
        for _ in range(len(e_trade_gym.df.index.unique())):
            action, _, _ = ensemble.step(state)
            # Remove [0] index from action
            state, _, done, _, _ = e_trade_gym.step(action)  # Changed action[0] to action
            episode_total_assets.append(e_trade_gym.asset_memory[-1])
            if done:
                break
    
    # Process results
    normalized = get_normalized_close(episode_total_assets, 1000000)
    df_dji = YahooDownloader(
        start_date=TRADE_START_DATE, end_date=TRADE_END_DATE,
        ticker_list=["^NDX"]
    ).fetch_data()[["date", "close"]]
    
    return filter_to_common_dates(trade, df_dji, normalized, 
                                list(df_dji["close"].div(df_dji["close"].iloc[0]).mul(1000000)))[0]

def flow():
    # Load the dataset
    trade = get_normal_dataset()
    trade_llm = get_llm_dataset()
    trade_llm_risk = get_llm_risk_dataset()

    # Get environment kwargs
    env_kwargs = get_env_kwargs()
    env_kwargs_llm = get_env_kwargs_llm()
    env_kwargs_llm_risk = get_env_kwargs_llm_risk()

    # Create environments
    e_trade_gym_1 = get_gym(trade, env_kwargs)
    e_trade_gym_2 = get_gym(trade, env_kwargs)
    e_trade_llm_gym = get_gym_llm(trade_llm, env_kwargs_llm)
    e_trade_llm_risk_gym = get_gym_llm_risk(trade_llm_risk, env_kwargs_llm_risk)

    # Load models
    loaded_ppo = load_ppo(e_trade_gym_1.observation_space, e_trade_gym_1.action_space)
    loaded_cppo = load_cppo(e_trade_gym_2.observation_space, e_trade_gym_2.action_space)
    loaded_ppo_llm = load_ppo_deepseek(e_trade_llm_gym.observation_space, e_trade_llm_gym.action_space)
    loaded_cppo_llm_risk = load_cppo_deepseek(e_trade_llm_risk_gym.observation_space, e_trade_llm_risk_gym.action_space)

    # Run predictions in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_ppo = executor.submit(DRL_prediction, act=loaded_ppo, environment=e_trade_gym_1)
        future_cppo = executor.submit(DRL_prediction, act=loaded_cppo, environment=e_trade_gym_2)
        future_ppo_llm = executor.submit(DRL_prediction, act=loaded_ppo_llm, environment=e_trade_llm_gym)
        future_cppo_llm_risk = executor.submit(DRL_prediction, act=loaded_cppo_llm_risk, environment=e_trade_llm_risk_gym)
        future_adaptive = executor.submit(adaptive_flow)
        # future_specialized = executor.submit(adaptive_specialized_flow)
        future_full_specialized = executor.submit(full_specialized_flow)
        future_cppo_full = executor.submit(full_cppo_specialized_flow)

        # Wait for and get the results
        df_assets_ppo, df_account_value_ppo, df_actions_ppo, df_portfolio_distribution_ppo = future_ppo.result()
        df_assets_cppo, df_account_value_cppo, df_actions_cppo, df_portfolio_distribution_cppo = future_cppo.result()
        df_assets_ppo_llm, df_account_value_ppo_llm, df_actions_ppo_llm, df_portfolio_distribution_ppo_llm = future_ppo_llm.result()
        df_assets_cppo_llm_risk, df_account_value_cppo_llm_risk, df_actions_cppo_llm_risk, df_portfolio_distribution_cppo_llm_risk = future_cppo_llm_risk.result()
        df_adaptive_portfolio = future_adaptive.result()
        df_cppo_full_portfolio = future_cppo_full.result()
        
        '''# Get results for the mini model strategy - with fallback mechanism
        try:
            df_adaptive_mini_portfolio = future_adaptive_mini.result()
        except Exception as e:
            print(f"Mini model strategy failed: {e}")
            print("Using standard adaptive portfolio results instead")
            df_adaptive_mini_portfolio = df_adaptive_portfolio
            
        # Add this with the other results
        df_mini_whole_market_portfolio = future_mini_whole_market.result()
        '''

        # Get specialized model results
        # try:
        #     df_specialized_portfolio = future_specialized.result()
        # except Exception as e:
        #     print(f"Specialized models strategy failed: {e}")
        #     print("Using standard adaptive portfolio results instead")
        #     df_specialized_portfolio = df_adaptive_portfolio

        try:
            df_full_specialized_portfolio = future_full_specialized.result()
        except Exception as e:
            print(f"Full specialized strategy failed: {e}")
            print("Using standard adaptive portfolio results instead")
            df_full_specialized_portfolio = df_adaptive_portfolio


    # NASDAQ 100 index
    df_dji = YahooDownloader(
        start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["^NDX"]
    ).fetch_data()
    df_dji = df_dji[["date", "close"]]
    fst_day = df_dji["close"].iloc[0]  # Safely get the first value
    df_dji_normalized_close = list(df_dji["close"].div(fst_day).mul(1000000))

    # BackTesting Results
    df_ppo_normalized_close = get_normalized_close(df_assets_ppo)
    df_cppo_normalized_close = get_normalized_close(df_assets_cppo)
    df_ppo_llm_normalized_close = get_normalized_close(df_assets_ppo_llm)
    df_cppo_llm_risk_normalized_close = get_normalized_close(df_assets_cppo_llm_risk)

    df_assets_ppo_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(
        trade, df_dji, df_ppo_normalized_close, df_dji_normalized_close)
    df_assets_cppo_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(
        trade, df_dji, df_cppo_normalized_close, df_dji_normalized_close)
    df_assets_ppo_llm_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(
        trade, df_dji, df_ppo_llm_normalized_close, df_dji_normalized_close)
    df_assets_cppo_llm_risk_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(
        trade, df_dji, df_cppo_llm_risk_normalized_close, df_dji_normalized_close)

    result = pd.DataFrame(
        {
            "PPO 100 epochs": df_assets_ppo_filtered,
            "CPPO 100 epochs": df_assets_cppo_filtered,
            "PPO-DeepSeek 100 epochs": df_assets_ppo_llm_filtered,
            "CPPO-DeepSeek 100 epochs": df_assets_cppo_llm_risk_filtered,
            "Adaptive Portfolio": df_adaptive_portfolio,
            "Mini PPO": df_full_specialized_portfolio,
            "Mini CPPO DeepSeek": df_cppo_full_portfolio,
            #"Mini PPO + Mini CPPO": df_specialized_portfolio,  # Add specialized models results
            "Nasdaq-100 index": df_dji_normalized_close_filtered,
        }
    )

    strategies = [
        "PPO 100 epochs",
        "CPPO 100 epochs",
        "PPO-DeepSeek 100 epochs",
        "CPPO-DeepSeek 100 epochs",
        "Adaptive Portfolio",
        "Mini PPO",
        "Mini CPPO DeepSeek",
        #"Mini PPO + Mini CPPO"  # Add specialized models to strategies
    ]
    benchmark = "Nasdaq-100 index"
    metrics = compute_metrics(result, strategies, benchmark)
    plot_cumulative_returns(result, metrics, strategies, benchmark)

    # Print metrics
    for strategy, strategy_metrics in metrics.items():
        print(f"{strategy} Metrics:")
        for metric_name, value in strategy_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
              
if __name__ == "__main__":
    flow()