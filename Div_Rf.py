import numpy as np
import pandas as pd
import datetime
from scipy.interpolate import interp1d
class Div_Rf:
    def __init__(self, T, dt):
        self.n_step = int(T / dt)
        self.rf_arr = pd.read_excel('Rates and DY.xlsx', sheet_name='RFR').to_numpy()

    def trading_days_in_between(self, y):
        y = y.date()
        days = np.busday_count(datetime.date(2023, 11, 17), y)
        return days

    def CubicSplineInterpolationDYRFR(self, time_step, a, b):
        first_value = b[0]
        last_value = b[-1]
        cs = interp1d(a, b, kind='cubic', fill_value=(first_value,last_value), bounds_error=False)
        return float(cs(time_step))

    def get_div_divdate(self, index):
        df = pd.read_excel('Rates and DY.xlsx', sheet_name=index)
        rows = len(df.axes[0])
        cols = len(df.axes[1])
        arr = df.to_numpy()
        div = [arr[x][1] for x in range(1, rows)]
        divdate = [self.trading_days_in_between(arr[x][0]) for x in range(1, rows)]
        return div, divdate

    def get_dividend(self, index):
        div, divdate = self.get_div_divdate(index)
        return [self.CubicSplineInterpolationDYRFR(i, divdate, div) for i in range(1, self.n_step + 1)]

    def get_all_dividend(self):
        return [self.get_dividend("HSCEID"), self.get_dividend("KOSPID"), self.get_dividend("SPXD")]

    def get_r_rdate(self, country):
        if country == "KR":
            return [self.rf_arr[x][5] / 100 for x in range(0, 8)], [self.rf_arr[x][4] for x in range(0, 8)]
        elif country == "US":
            return [self.rf_arr[x][1] / 100 for x in range(0, 11)], [self.rf_arr[x][0] for x in range(0, 11)]
        elif country == "HK":
            return [self.rf_arr[x][3] / 100 for x in range(0, 8)], [self.rf_arr[x][2] for x in range(0, 8)]

    def get_rf(self, country):
        r, rdate = self.get_r_rdate(country)
        return [self.CubicSplineInterpolationDYRFR(i, rdate, r) for i in range(1, self.n_step + 1)]

    def get_all_rf(self):
        return [self.get_rf("HK"), self.get_rf("KR"), self.get_rf("US")]

    def get_forward_rates(self, r, dt):
        r_t = r[:-1]  # risk-free rates excluding the last day
        r_t_plus_dt = r[1:]  # risk-free rates excluding the first day
        forward_rates = [((1 + r_t_plus_dt[i] / 252) ** (i + 1) / (1 + r_t[i] / 252) ** i - 1) / dt for i in
                         range(len(r_t))]
        return [r[0]] + forward_rates  # prepend the 1-day rate
