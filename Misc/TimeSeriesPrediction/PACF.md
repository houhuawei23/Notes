### Partial Autocorrelation Function (PACF)

The Partial Autocorrelation Function (PACF) is a statistical tool used in time series analysis to measure the correlation between a time series and its lagged versions, after removing the effects of shorter lags. In other words, the PACF helps to isolate the relationship between the time series and a specific lag, controlling for the influence of other lags.

#### Key Concepts:

1. **Autocorrelation**: As discussed earlier, autocorrelation measures the correlation between a time series and its lagged versions. However, this correlation can be influenced by other lags.
2. **Partial Autocorrelation**: The partial autocorrelation at lag \( k \) (denoted as \( \phi_{kk} \)) is the correlation between the time series and its \( k \)-th lag, after removing the effects of all shorter lags (i.e., lags 1 through \( k-1 \)).
3. **PACF Plot**: The PACF plot shows the partial autocorrelation coefficients for different lags on the y-axis and the lags on the x-axis. It helps in identifying the significant lags that are not explained by shorter lags.

#### Mathematical Definition:

The partial autocorrelation at lag \( k \) is the coefficient \( \phi_{kk} \) in the following autoregressive model:

\[
X_t = \phi_{k1} X_{t-1} + \phi_{k2} X_{t-2} + \cdots + \phi_{kk} X_{t-k} + \epsilon_t
\]

where:

- \( X_t \) is the value of the time series at time \( t \).
- \( X_{t-k} \) is the value of the time series at time \( t-k \) (i.e., \( k \) time periods earlier).
- \( \epsilon_t \) is the error term at time \( t \).
- \( \phi_{kk} \) is the partial autocorrelation coefficient at lag \( k \).

#### Interpretation:

- **Lag 0 (k=0)**: The partial autocorrelation at lag 0 is always 1 because it represents the correlation of the time series with itself, which is perfect.
- **Positive Partial Autocorrelation**: If \( \phi_{kk} > 0 \), it indicates that the time series values at lag \( k \) are positively correlated after controlling for shorter lags.
- **Negative Partial Autocorrelation**: If \( \phi_{kk} < 0 \), it indicates that the time series values at lag \( k \) are negatively correlated after controlling for shorter lags.
- **Zero Partial Autocorrelation**: If \( \phi_{kk} \approx 0 \), it indicates that there is no significant linear relationship between the time series and its lagged version at lag \( k \) after controlling for shorter lags.

#### Example:

Consider a simple time series representing the monthly sales of a product over a year:

\[
\text{Sales} = \{100, 120, 110, 130, 140, 150, 160, 170, 180, 190, 200, 210\}
\]

To calculate the PACF at lag 2, we need to fit an autoregressive model of order 2 and extract the coefficient \( \phi_{22} \):

1. **Fit the AR(2) Model**:
   \[
   X_t = \phi_{21} X_{t-1} + \phi_{22} X_{t-2} + \epsilon_t
   \]
   Using the sales data, estimate the coefficients \( \phi_{21} \) and \( \phi_{22} \).
2. **Extract \( \phi_{22} \)**:
   The coefficient \( \phi_{22} \) represents the partial autocorrelation at lag 2, controlling for the effect of lag 1.

#### Visual Representation:

The PACF is often visualized using a plot called the **Partial Autocorrelation Plot** or **PACF Plot**. This plot shows the partial autocorrelation coefficients for different lags on the y-axis and the lags on the x-axis. The plot typically includes a dashed line representing the significance level, which helps in determining whether the partial autocorrelation coefficients are statistically significant.

### Conclusion:

The Partial Autocorrelation Function (PACF) is a crucial tool in time series analysis, particularly in identifying the order of the autoregressive (AR) component in ARIMA models. By examining the PACF, analysts can determine which lags are significant after controlling for the effects of shorter lags, which is essential for building accurate forecasting models.
