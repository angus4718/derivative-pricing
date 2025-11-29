import numpy as np
import pandas as pd
import datetime
from scipy.interpolate import interp1d


class Div_Rf:
    def __init__(self, T, dt):
        self.n_step = int(T / dt)
        self.rf_df = pd.read_excel("data/rates.xlsx", sheet_name="RFR")
        self.rf_arr = self.rf_df.to_numpy()
        self.trading_days_per_year = 252
        self.base_date = datetime.date(2023, 11, 17)

    def trading_days_in_between(self, y):
        days = np.busday_count(self.base_date, y.date())
        return days

    def CubicSplineInterpolation(self, time_step, a, b):
        first_value = b[0]
        last_value = b[-1]
        cs = interp1d(
            a, b, kind="cubic", fill_value=(first_value, last_value), bounds_error=False
        )
        return float(cs(time_step))

    def get_div_divdate(self, index):
        df = pd.read_excel("data/rates.xlsx", sheet_name=index)
        rows = len(df.axes[0])
        arr = df.to_numpy()
        div = [arr[x][1] for x in range(1, rows)]
        divdate = [self.trading_days_in_between(arr[x][0]) for x in range(1, rows)]
        return div, divdate

    def get_dividend(self, index):
        div, divdate = self.get_div_divdate(index)
        return [
            self.CubicSplineInterpolation(i, divdate, div)
            for i in range(1, self.n_step + 1)
        ]

    def get_all_dividend(self):
        return [
            self.get_dividend("HSCEID"),
            self.get_dividend("KOSPID"),
            self.get_dividend("SPXD"),
        ]

    def get_r_rdate(self, country):
        # Determine columns (keep previous fixed mapping)
        if country == "KR":
            date_col, rate_col = 4, 5
        elif country == "US":
            date_col, rate_col = 0, 1
        elif country == "HK":
            date_col, rate_col = 2, 3
        else:
            raise ValueError(f"Unsupported country code: {country}")

        dseries = self.rf_df.iloc[:, date_col]
        rseries = self.rf_df.iloc[:, rate_col]
        rconv = pd.to_numeric(rseries, errors="coerce")
        day_vals = pd.to_numeric(dseries, errors="coerce").astype("Float64")
        df_pairs = pd.DataFrame({"day": day_vals, "rate": rconv})
        df_pairs = df_pairs.dropna(subset=["day", "rate"]).reset_index(drop=True)
        df_pairs["day"] = df_pairs["day"].astype(int)

        # Aggregate duplicate days (if multiple rows have same date) by averaging rates
        df_agg = df_pairs.groupby("day", as_index=False).agg({"rate": "mean"})

        # Convert rates from percent to decimal
        rates = (df_agg["rate"].astype(float) / 100.0).tolist()
        rdates = df_agg["day"].astype(int).tolist()

        return rates, rdates

    def get_rf(self, country):
        r, rdate = self.get_r_rdate(country)
        return [
            self.CubicSplineInterpolation(i, rdate, r)
            for i in range(1, self.n_step + 1)
        ]

    def get_all_rf(self):
        return [self.get_rf("HK"), self.get_rf("KR"), self.get_rf("US")]

    def get_forward_rates(self, r, dt):
        tdpy = self.trading_days_per_year
        r_t = r[:-1]
        r_t_plus_dt = r[1:]
        forward_rates = [
            ((1 + r_t_plus_dt[i] / tdpy) ** (i + 1) / (1 + r_t[i] / tdpy) ** i - 1) / dt
            for i in range(len(r_t))
        ]
        return [r[0]] + forward_rates
