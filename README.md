# LTC-vs-LSTM

# LTC vs. LSTM Comparative Study on Beijing PM2.5 Dataset

## Overview
This project involves a comparative analysis of **Liquid Time-Constant (LTC)** neural networks and **Long Short-Term Memory (LSTM)** networks. Both models were implemented and trained on the Beijing PM2.5 air quality dataset, aiming to evaluate their effectiveness in time-series forecasting tasks, specifically air pollution (PM2.5) predictions.

## Dataset Description
- **Dataset**: Beijing PM2.5 Dataset
- **Time Span**: January 1, 2010, to December 31, 2014
- **Features Included:** PM2.5 concentration, temperature, humidity, wind speed, and additional meteorological factors.
- **Purpose of Choosing Dataset:**
  - Complex temporal dynamics with hourly granularity.
  - Real-world noisy data making it suitable for evaluating robustness and predictive stability.



### LTC vs. LSTM Results
| Metric | LTC | LSTM |
|--------|-----|------|
| **MSE**  | 675.55 | 584.20 |
| **MAE**  | 14.32 | 13.00 |
| **R²**   | 0.9289 | 0.9385 |

- **Interpretation:**
  - LSTM slightly outperformed LTC based on accuracy metrics.
  - LTC showed quicker convergence and smoother predictions, indicating robustness and better performance in noisy or irregular data environments.

## Future Work
- **Detailed Hyperparameter Optimization:** To further enhance LTC’s predictive performance.
- **Robustness Testing:** Compare models explicitly under varied noise conditions.
- **Computational Efficiency Analysis:** Measure training/inference speed and memory efficiency.
- **Hybrid Approaches:** Investigate hybrid architectures combining LTC and LSTM strengths.

This comparative analysis demonstrates clear trade-offs between LTC and LSTM, emphasizing considerations beyond accuracy alone, such as robustness and computational efficiency.

