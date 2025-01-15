import numpy as np
import math
from Vol_Calculation import Vol_Calculation
import datetime as datetime
import os
np.random.seed(42)

T = 3
dt = 1 / 252
n_sim = 1000
spot_prices = [5974.47, 331.63, 4513.91]
today_date=datetime.datetime(2023, 11, 17)
vol_calc = Vol_Calculation(spot_prices, T, dt, today_date)
vol_calc.load_data('Data4150.xlsx', ['HSCEI', 'KOSPI2', 'SPX'])
implied_vol_dfs, exercise_dates = vol_calc.excel_to_implied_vol_dfs('Data4150.xlsx', ['HSCEI', 'KOSPI2', 'SPX'])
params_dfs = vol_calc.implied_vol_curve_fitting(implied_vol_dfs)

class MonteCarlo:
    def __init__(self, r, r_forward, d_forward, T, dt, n_sim):
        # Parameters
        self.T = T
        self.dt = dt
        self.n_step = int(self.T / self.dt)  # if dt is daily, 252 * 3 = 756
        self.n_sim = n_sim
        self.obs_duration = int(1 / self.dt / 2)  # if dt is daily, 252 / 2 = 126
        self.note_denomination = 10000
        self.min_coupon = 0.0001
        self.cor_matrix = np.array([[1, 0.551, 0.111],
                                    [0.551, 1, 0.191],
                                    [0.111, 0.191, 1]])
        self.r_forward = r_forward
        self.d_forward = d_forward
        self.r = r

        # Vol_Calculation
        self.spot_prices = [5974.47, 331.63, 4513.91]
        self.today_date = datetime.datetime(2023, 11, 17)
        self.vol_calc = Vol_Calculation(spot_prices, T, dt, today_date)
        self.vol_calc.load_data('Data4150.xlsx', ['HSCEI', 'KOSPI2', 'SPX'])
        self.implied_vol_dfs, self.exercise_dates = self.vol_calc.excel_to_implied_vol_dfs('Data4150.xlsx', ['HSCEI', 'KOSPI2', 'SPX'])
        self.params_dfs = vol_calc.implied_vol_curve_fitting(self.implied_vol_dfs)

        # Pre-compute sigma values
        self.start_value = 0.01
        self.end_value = 10
        self.step = 0.01
        self.x_values = np.log(np.arange(self.start_value, self.end_value, self.step))
        self.sigma_values = self.precompute_sigma()

        self.W_1_half = np.random.standard_normal((int(self.n_sim / 2), self.n_step))
        self.W_2_half = np.random.standard_normal((int(self.n_sim / 2), self.n_step))
        self.W_3_half = np.random.standard_normal((int(self.n_sim / 2), self.n_step))

    def precompute_sigma(self):
        # Check if the file exists
        if os.path.exists('sigma.npy'):
            try:
                # Try to load pre-computed sigma
                return np.load('sigma.npy')
            except Exception as e:
                # If loading fails for any reason, print the error and continue to compute sigma
                print(f"Failed to load pre-computed Sigma values: {e}")

        # Compute sigma values
        n_days = self.n_step
        n_indices = 3
        sigma_values = np.empty((len(self.x_values), n_days, n_indices))
        for i, x in enumerate(self.x_values):
            for day in range(n_days):
                for index_no in range(n_indices):
                    sigma_values[i][day][index_no] = self.vol_calc.interpolated_local_vol(x, index_no, day,
                                                                                          self.params_dfs)

        print('Precompute Sigma Done')

        # Save computed sigma values
        try:
            np.save('sigma.npy', sigma_values)
            print('Saved Sigma values to file.')
        except Exception as e:
            print(f"Failed to save pre-computed Sigma values: {e}")

        return sigma_values
    def price(self, coupon):
        L = np.linalg.cholesky(self.cor_matrix)
        W_1_half = self.W_1_half
        W_2_half = self.W_2_half
        W_3_half = self.W_3_half
        Z_1_half = W_1_half
        Z_2_half = W_1_half * L[1, 0] + W_2_half * L[1, 1]
        Z_3_half = W_1_half * L[2, 0] + W_2_half * L[2, 1] + W_3_half * L[2, 2]
        Z_1 = np.concatenate((Z_1_half, -Z_1_half), axis=0)
        Z_2 = np.concatenate((Z_2_half, -Z_2_half), axis=0)
        Z_3 = np.concatenate((Z_3_half, -Z_3_half), axis=0)

        # Simulate moneyness path of Kospi 200, S&P 500, and HSCEI
        S_1 = np.empty((self.n_sim, self.n_step + 1))
        S_2 = np.empty((self.n_sim, self.n_step + 1))
        S_3 = np.empty((self.n_sim, self.n_step + 1))
        for i in range(self.n_sim):
            S_1[i][0] = 1
            S_2[i][0] = 1
            S_3[i][0] = 1
            for j in range(1, self.n_step + 1):
                # Simulate Price Path
                sigma = []
                sigma.append(self.get_sigma(S_1[i][j - 1], j - 1, 0))
                sigma.append(self.get_sigma(S_2[i][j - 1], j - 1, 0))
                sigma.append(self.get_sigma(S_3[i][j - 1], j - 1, 0))
                S_1[i][j] = S_1[i][j - 1] * math.exp(
                    (self.r_forward[0][j - 1] - self.d_forward[0][j - 1] - 0.5 * sigma[0] ** 2) * self.dt + sigma[0] * Z_1[i][
                        j - 1] * math.sqrt(self.dt))
                S_2[i][j] = S_2[i][j - 1] * math.exp(
                    (self.r_forward[1][j - 1] - self.d_forward[1][j - 1] - 0.5 * sigma[1] ** 2) * self.dt + sigma[1] * Z_2[i][
                        j - 1] * math.sqrt(self.dt))
                S_3[i][j] = S_3[i][j - 1] * math.exp(
                    (self.r_forward[2][j - 1] - self.d_forward[2][j - 1] - 0.5 * sigma[2] ** 2) * self.dt + sigma[2] * Z_3[i][
                        j - 1] * math.sqrt(self.dt))

        # Observation
        obsS_1 = S_1[:, self.obs_duration::self.obs_duration]
        obsS_2 = S_2[:, self.obs_duration::self.obs_duration]
        obsS_3 = S_3[:, self.obs_duration::self.obs_duration]
        laggard = np.minimum.reduce((obsS_1, obsS_2, obsS_3))
        knock_in = [min(i) <= 0.5 for i in laggard]
        obs_dates = np.linspace(0.5, self.T, int(self.T / 0.5))  # Semi-annual observation dates
        discount_factors = np.array([np.exp(-self.r[2][int(self.n_step * obs_date / 3) - 1] * obs_date) for obs_date in obs_dates])  # Discount factors for each observation date
        payoffs = []
        for i in range(self.n_sim):
            total_pay = 0
            for j, discount_factor in enumerate(discount_factors):
                # No knock-out
                if laggard[i][j] < 1:
                    total_pay += discount_factor * self.note_denomination * self.min_coupon
                # Knock-out event
                else:
                    total_pay += discount_factor * self.note_denomination * (1 + (j + 1) * coupon)
                    break  # Exit the loop as the note is redeemed early

                # Final redemption (if no knock-out occurs before maturity)
                if j == len(discount_factors) - 1:
                    if knock_in[i]:
                        total_pay += discount_factor * self.note_denomination * min(1, laggard[i][j])
                    else:
                        total_pay += discount_factor * self.note_denomination

            payoffs.append(total_pay)
        return np.mean(payoffs)

    def bisection(self, price, tol=1):
        coupon_l = 0
        coupon_r = 0.01

        def error(coupon):
            return self.price(coupon) - price

        while error(coupon_r) <= 0:
            coupon_r += 0.01
        print(coupon_l, coupon_r)
        coupon_temp = (coupon_l + coupon_r) / 2
        while abs(error(coupon_temp)) >= tol:
            if error(coupon_temp) > 0:
                coupon_r = coupon_temp
            else:
                coupon_l = coupon_temp
            coupon_temp = (coupon_l + coupon_r) / 2
            print(error(coupon_temp), coupon_temp)
        return coupon_temp

    def get_sigma(self, price, day, index_no):
        x_0 = price
        x_index = int((x_0 - self.start_value) / self.step)
        return self.sigma_values[x_index][day][index_no]