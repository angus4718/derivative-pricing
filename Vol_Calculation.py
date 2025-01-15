import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.misc import derivative
from Div_Rf import Div_Rf
from scipy.interpolate import interp1d

class Vol_Calculation:
    def __init__(self, spot_prices, T, dt, today_date):
        self.spot_prices = spot_prices
        self.T = T
        self.dt = dt
        self.today_date = today_date
        self.div_rf = Div_Rf(T, dt)
        self.r = self.div_rf.get_all_rf()
        self.d = self.div_rf.get_all_dividend()
        self.params_dfs = []
        self.exercise_dates = []
        self.implied_vol_dfs = []

    def daysinbetween3(self, y):
        y = y.date()
        days = np.busday_count(self.today_date.date(), y)
        return days

    def implied_vol_curve_func(self, x, sigma_atm, delta, gamma, kappa):
        return np.max(sigma_atm ** 2 + delta * (np.tanh(kappa * x) / kappa) + gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2,0)

    def implied_vol_curve_fitting(self, implied_vol_dfs):
        params_dfs = []
        for i in range(3):
            df = implied_vol_dfs[i]
            params_df = pd.DataFrame(columns=['sigma_atm', 'delta', 'gamma', 'kappa'])
            for column in df.columns:
                x = np.log(df.index.values / self.spot_prices[i])
                y = df[column].values ** 2
                mask = ~np.isnan(y)
                x = x[mask]
                y = y[mask]
                if x.size > 0 and y.size > 0:
                    def residuals(params):
                        sigma_atm, delta, gamma, kappa = params
                        if delta ** 2 - 2 * gamma * sigma_atm ** 2 >= 0:
                            return 1e20
                        return np.sum((y - self.implied_vol_curve_func(x, *params)) ** 2)

                    constraint_met = False
                    while not constraint_met:
                        initial_guess = np.random.rand(4)  # Randomize initial guess
                        result = minimize(residuals, initial_guess, method='SLSQP')
                        sigma_atm, delta, gamma, kappa = result.x
                        if delta ** 2 - 2 * gamma * sigma_atm ** 2 < 0 and sigma_atm > 0 and sigma_atm < 0.3 and gamma > 0:  # Check if constraint is met
                            constraint_met = True
                            params_df.loc[column] = result.x

            params_dfs.append(params_df)
        return params_dfs

    def excel_to_implied_vol_dfs(self, excel_file, indices_names):
        xls = pd.ExcelFile(excel_file)
        implied_vol_dfs = []
        exercise_dates = []
        for sheet in indices_names:
            if sheet in xls.sheet_names:
                df = xls.parse(sheet, index_col=0).replace(0, np.nan)
                implied_vol_dfs.append(df)
                exercise_dates.append(df.columns.tolist())
        return implied_vol_dfs, exercise_dates

    def load_data(self, excel_file, indices_names):
        self.implied_vol_dfs, self.exercise_dates = self.excel_to_implied_vol_dfs(excel_file, indices_names)

    def implied_to_local_vol(self, x_0, index_no, params_dfs, exercise_date):
        sigma_atm = params_dfs[index_no].loc[exercise_date, 'sigma_atm']
        delta = params_dfs[index_no].loc[exercise_date, 'delta']
        gamma = params_dfs[index_no].loc[exercise_date, 'gamma']
        kappa = params_dfs[index_no].loc[exercise_date, 'kappa']

        def w(x):
            return self.implied_vol_curve_func(x, sigma_atm, delta, gamma, kappa) * self.daysinbetween3(exercise_date)

        y = x_0 - (sum(self.r[index_no][:self.daysinbetween3(exercise_date)]) - sum(self.d[index_no][:self.daysinbetween3(exercise_date)])) * self.dt

        dw_dy = derivative(w, x_0, dx=1e-6)
        d2w_dy2 = derivative(w, x_0, dx=1e-6, n=2)

        local_var = self.implied_vol_curve_func(x_0, sigma_atm, delta, gamma, kappa) / (1 - y / w(x_0) * dw_dy + .25 * (-.25 - 1 / w(x_0) + (y / w(x_0)) ** 2) * (dw_dy) ** 2 + .5 * d2w_dy2)

        local_vol = np.sqrt(local_var)
        return local_vol

    def interpolated_local_vol(self, x_0, index_no, day, params_dfs):
        local_vol_list = []
        day_list = []
        for exercise_date in self.exercise_dates[index_no]:
            spot_moneyness = x_0 + (sum(self.r[index_no][:day]) - sum(self.d[index_no][:day])) * self.dt
            y = spot_moneyness - (sum(self.r[index_no][:self.daysinbetween3(exercise_date)]) - sum(self.d[index_no][:self.daysinbetween3(exercise_date)])) * self.dt
            local_vol_value = self.implied_to_local_vol(y, index_no, params_dfs, exercise_date)
            if exercise_date == self.exercise_dates[index_no][0]:
                first_value = local_vol_value
            elif exercise_date == self.exercise_dates[index_no][-1]:
                last_value = local_vol_value
            if not np.isnan(local_vol_value):
                local_vol_list.append(local_vol_value)
                day_list.append(self.daysinbetween3(exercise_date))

        f = interp1d(day_list, local_vol_list, kind='cubic', fill_value=(first_value,last_value), bounds_error=False)

        return f(day)