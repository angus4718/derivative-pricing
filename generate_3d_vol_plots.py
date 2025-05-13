import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from Vol_Calculation import Vol_Calculation
import os

# Load data and calculate volatility
T = 3
dt = 1 / 252
spot_prices = [5974.47, 331.63, 4513.91]
today_date = pd.Timestamp("2023-11-17")
vol_calc = Vol_Calculation(spot_prices, T, dt, today_date)

# Load implied volatility data and exercise dates
vol_calc.load_data('Data4150.xlsx', ['HSCEI', 'KOSPI2', 'SPX'])
implied_vol_dfs, exercise_dates = vol_calc.excel_to_implied_vol_dfs('Data4150.xlsx', ['HSCEI', 'KOSPI2', 'SPX'])

# Fit parameters for implied volatility curves
params_dfs = vol_calc.implied_vol_curve_fitting(implied_vol_dfs)

# Load precomputed sigma values
def load_precomputed_sigma():
    if os.path.exists('sigma.npy'):
        try:
            sigma_values = np.load('sigma.npy')
            print("Successfully loaded precomputed sigma values.")
            return sigma_values
        except Exception as e:
            print(f"Failed to load precomputed sigma values: {e}")
            raise
    else:
        raise FileNotFoundError("sigma.npy file not found. Please precompute sigma values first.")

sigma_values = load_precomputed_sigma()

# Function to generate 3D plots
def plot_3d_volatility(x_axis, y_axis, z_axis, title, xlabel, ylabel, zlabel, cmap=cm.viridis):
    """
    Create a 3D plot for volatility data.

    Args:
        x_axis (np.ndarray): X-axis values (e.g., strike prices or moneyness).
        y_axis (np.ndarray): Y-axis values (e.g., exercise dates or time to maturity).
        z_axis (np.ndarray): Corresponding Z-axis values (volatility). Must have shape (len(y_axis), len(x_axis)).
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        zlabel (str): Label for the Z-axis.
        cmap (colormap): Colormap for the surface plot.

    Returns:
        None
    """
    # Ensure X, Y, and Z have matching shapes
    X, Y = np.meshgrid(x_axis, y_axis)  # Create grid from x_axis (strikes) and y_axis (maturities)
    Z = z_axis

    if Z.shape != X.shape:
        raise ValueError(f"Shape mismatch: X/Y shape is {X.shape}, Z shape is {Z.shape}")

    # Create figure and 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='k', alpha=0.8)

    # Add labels, title, and color bar
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()

# Generate 3D plots for each underlying
underlying_names = ['HSCEI', 'KOSPI2', 'SPX']
for index_no, name in enumerate(underlying_names):
    # Generate implied volatility data
    implied_vol_df = implied_vol_dfs[index_no]
    strikes = implied_vol_df.index.values
    maturities = np.array([vol_calc.daysinbetween3(pd.Timestamp(date)) for date in implied_vol_df.columns])
    implied_vol_data = implied_vol_df.values

    # Plot implied volatility
    plot_3d_volatility(
        x_axis=strikes,
        y_axis=maturities,
        z_axis=implied_vol_data.T,  # Transpose implied vol data if needed
        title=f"Implied Volatility - {name}",
        xlabel="Strike Price",
        ylabel="Days to Maturity",
        zlabel="Implied Volatility",
        cmap=cm.plasma,
    )

    # Generate local volatility data from precomputed sigmas
    local_vol_data = sigma_values[:len(maturities), :len(strikes), index_no]  # Extract local vol for the index

    # Check if transpose is needed
    if local_vol_data.shape != (len(maturities), len(strikes)):
        local_vol_data = local_vol_data.T  # Transpose if dimensions are flipped

    # Plot local volatility
    plot_3d_volatility(
        x_axis=strikes,
        y_axis=maturities,
        z_axis=local_vol_data,  # Correct shape
        title=f"Local Volatility - {name}",
        xlabel="Strike Price",
        ylabel="Days to Maturity",
        zlabel="Local Volatility",
        cmap=cm.viridis,
    )