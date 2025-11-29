import matplotlib.pyplot as plt
from div_rfr import Div_Rf


def plot_data(r, d):
    # Plotting all lists in r
    for i, r_list in enumerate(r):
        plt.figure(figsize=(10, 6))
        plt.plot(r_list)
        plt.title(f"Risk-free Rate Series {i+1}")
        plt.xlabel("Days")
        plt.ylabel("Risk-free rate")
        plt.show()

    # Plotting all lists in d
    for i, d_list in enumerate(d):
        plt.figure(figsize=(10, 6))
        plt.plot(d_list)
        plt.title(f"Dividend Yield Series {i+1}")
        plt.xlabel("Days")
        plt.ylabel("Dividend yield")
        plt.show()


# Configurations
T = 3
dt = 1 / 252
n_sim = 1000

div_rf = Div_Rf(T, dt)
r = div_rf.get_all_rf()
d = div_rf.get_all_dividend()
r_forward = [div_rf.get_forward_rates(index, dt) for index in r]
d_forward = [div_rf.get_forward_rates(index, dt) for index in d]

plot_data(r_forward, d_forward)
