# Optimizing Coupon Rate for a Step-Up Autocallable Note Using Monte Carlo Simulation

This repository implements a Monte Carlo framework to determine the optimal coupon rate for a step-up autocallable note. It uses cubic spline interpolation, Dupire’s local volatility model, and a bisection method to match the simulated price to 98% of the issue price.

---

## Overview

The note is linked to three indices: HSCEI, Kospi 200, and S&P 500. The engine:
- Simulates correlated index paths with local volatility.
- Interpolates risk-free rates and dividend yields via cubic splines.
- Uses bisection to find the coupon rate that prices the note at 98% of par.

---

## Note Structure

- Underlyings:
  - HSCEI (Spot 5974.47)
  - Kospi 200 (Spot 331.63)
  - S&P 500 (Spot 4513.91)
- Trade date: 17/11/2023
- Maturity: 3 years
- Denomination: USD 10,000
- Issue price: 100%

Coupon Payments and Knock-Out Event:
- Minimum coupon: 0.01% per annum (paid semi-annually unless a knock-out event occurs).
- Knock-out event: Triggered when the laggard index (lowest performer) closes at or above the initial spot price. The note is redeemed early.

Final Redemption:
- Knock-In Event: Triggered if the laggard index's closing price falls to 50% or less of the initial spot price.
- Redemption Scenarios:
  - No knock-in event: Full denomination redeemed.
  - Knock-in event: Redeemed at the lesser of 100% or the laggard index's closing price ratio to its initial spot price.

---

## Methodology

- Risk-free rates and dividends:
  - Spot curves interpolated via cubic splines; forwards derived for simulation.
- Volatility:
  - Implied vols fitted; transformed to local vol with Dupire’s model.
- Monte Carlo:
  - Geometric Brownian Motion with local vol, correlated shocks, and antithetic variates.
  - Tracks knock-in/knock-out and computes payoffs.
- Pricing target:
  - Bisection over coupon rate to match 98% of issue price.

---

## Running the Simulation

1. Configure parameters in the script:
   - T = 3, dt = 1/252, n_sim = 1000, target price = 9800.
2. Run:
   ```bash
   python main.py
   ```
3. Output:
   - Optimal coupon rate (e.g., 13.37% for the sample config).
   - Optional plots: local vol surfaces and sample price paths.

---

## Code Structure

- Div_Rf:
  - Spline interpolation for risk-free and dividend curves; provides forward rates.
- Vol_Calculation:
  - Fits implied vol curves; builds and interpolates local vol surfaces.
- MonteCarlo:
  - Path simulation, event tracking, payoff calculation; bisection for coupon rate.