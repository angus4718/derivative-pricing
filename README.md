# Optimizing Coupon Rate for a Step-Up Autocallable Note Using Monte Carlo Simulation

This repository contains the implementation of a **Monte Carlo Simulation** to determine the optimal coupon rate for a **Step-Up Autocallable Note**. The project uses advanced financial modeling techniques, including **cubic spline interpolation**, **local volatility modeling**, and the **bisection method**, to match the simulated note price to 98% of the issue price.

---

## **Overview**

This project aims to optimize the coupon rate of a **step-up autocallable note**, a structured financial product based on three indices: **HSCEI**, **Kospi 200**, and **S&P 500**. 

### **Key Components**:
1. **Monte Carlo Simulation**: Simulates the price paths of the underlying indices and calculates the note price.
2. **Local Volatility Surface**: Converts implied volatilities to local volatilities using **Dupire's local volatility model**.
3. **Bisection Method**: Iteratively solves for the coupon rate that aligns the simulated note price with the target price.
4. **Risk-Free Rates and Dividend Yields**: Interpolates spot rates using **cubic spline interpolation**.

---

## **Note Structure**

### **Underlying Indices**
- **HSCEI** (Spot = 5974.47)
- **Kospi 200** (Spot = 331.63)
- **S&P 500** (Spot = 4513.91)

### **Key Features**:
- **Trade Date**: 17/11/2023  
- **Maturity**: 3 years after the trade date  
- **Denomination**: USD 10,000  
- **Issue Price**: 100% of the note denomination  

### **Coupon Payments and Knock-Out Event**:
- Minimum coupon: 0.01% per annum (paid semi-annually unless a knock-out event occurs).
- Knock-out event: Triggered when the laggard index (lowest performer) closes at or above the initial spot price. The note is redeemed early.

### **Final Redemption**:
- **Knock-In Event**: Triggered if the laggard index's closing price falls to 50% or less of the initial spot price.
- **Redemption Scenarios**:
  - No knock-in event: Full denomination redeemed.
  - Knock-in event: Redeemed at the lesser of 100% or the laggard index's closing price ratio to its initial spot price.

---

## **Features**

- **Monte Carlo Simulation**:
  - Simulates the price paths of the three indices using **Geometric Brownian Motion**.
  - Incorporates **correlated random variables** and antithetic variates for variance reduction.
- **Volatility Surface**:
  - Constructs a local volatility surface using implied volatilities.
  - Uses **Dupire's local volatility model** for conversion.
- **Risk-Free Rates and Dividend Yields**:
  - Interpolates using **cubic spline interpolation** with data from Bloomberg Terminal.
- **Bisection Method**:
  - Solves for the coupon rate that sets the simulated note price to 98% of the issue price.
- **Customizable Parameters**:
  - Total time to maturity (`T`), time step (`dt`), number of simulations (`n_sim`), and target price can be adjusted.

---

## **Setup**

### **Dependencies**
- Python 3.8+
- Required libraries:
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/step-up-autocallable-note.git
   cd step-up-autocallable-note
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How It Works**

### **1. Risk-Free Rates and Dividend Yields**
- Spot rates for interest rates and dividend yields are interpolated using **cubic spline interpolation**.
- Forward rates are calculated for use in the Monte Carlo simulation.

### **2. Local Volatility Surface**
- Implied volatilities are sourced from market data and fitted to a continuous curve.
- **Dupire’s local volatility model** transforms implied volatilities into local volatilities.

### **3. Monte Carlo Simulation**
- Simulates index price paths using **Geometric Brownian Motion**, incorporating:
  - Forward risk-free rates (`rt`) and dividend yields (`dt`).
  - Local volatilities (`σt`).
- Tracks knock-in and knock-out events and calculates the payoff for each simulation.

### **4. Bisection Method**
- Solves for the coupon rate using an iterative root-finding algorithm.
- Ensures that the simulated note price matches the target price (98% of the issue price).

---

## **Running the Simulation**

1. **Set Parameters**:
   Modify the configurations in the script:
   - Total time to maturity (`T` = 3).
   - Time step (`dt` = 1/252).
   - Number of simulations (`n_sim` = 1000).
   - Target price (`price` = 9800).

2. **Run the Simulation**:
   Execute the main script:
   ```bash
   python worlds_simulator.py
   ```

3. **Output**:
   - The script outputs the optimal coupon rate (`13.37%`) for the given parameters.
   - Visualization of volatility surfaces and simulated price paths.

---

## **Code Structure**

### **Main Classes**
- **`Div_Rf`**:
  - Interpolates risk-free rates and dividend yields.
  - Provides forward rates for Monte Carlo simulation.

- **`Vol_Calculation`**:
  - Fits implied volatility curves and constructs local volatility surfaces.
  - Interpolates local volatilities for specific dates and strike prices.

- **`MonteCarlo`**:
  - Implements Monte Carlo simulation for index price paths.
  - Tracks knock-in/knock-out events and calculates note payoffs.
  - Uses the **bisection method** to determine the optimal coupon rate.

### **Workflow**
1. **Risk-Free Rates and Dividend Yields**:
   - Compute forward rates using `Div_Rf`.
2. **Volatility Surface**:
   - Load implied volatility data and fit local volatility surfaces using `Vol_Calculation`.
3. **Monte Carlo Simulation**:
   - Simulate price paths and calculate note price using `MonteCarlo`.

---

## **Outputs**

- **Optimal Coupon Rate**: 13.37% for a note price of 98% of the issue price.
- **Volatility Surfaces**:
  - Smooth, fitted curves for implied and local volatilities.
- **Simulated Note Price**:
  - Average payoff across 1000 simulations.

---

## **Customization**

- Modify the following parameters in the script:
  - **Time to maturity**: `T`
  - **Number of simulations**: `n_sim`
  - **Target price**: `price`
  - **Indices**: Replace `HSCEI`, `Kospi 200`, and `S&P 500` with new underlying assets.
- Input new implied volatility or risk-free rate data for different markets.

---

## **Conclusion**

This project successfully demonstrates how to determine the coupon rate of a **step-up autocallable note** using advanced financial modeling techniques. By incorporating **Monte Carlo simulation**, **local volatility modeling**, and the **bisection method**, the model provides a robust framework for pricing complex financial instruments.

---

## **Author**
- Chun Hin (Angus) CHEUNG

---

## **License**

This project is licensed under the MIT License.
