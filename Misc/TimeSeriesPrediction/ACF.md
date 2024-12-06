### Autocorrelation Function (ACF)

The Autocorrelation Function (ACF) is a statistical tool used to measure the correlation between a time series and its lagged versions. In other words, it quantifies how similar a time series is to itself at different points in time. The ACF is widely used in time series analysis, particularly in the context of identifying patterns, trends, and seasonality, as well as in model building for forecasting.

#### Key Concepts:

1. **Lag**: The lag $k$ represents the time difference between the observations in the time series. For example, if you have a monthly time series, a lag of 1 means you are comparing each month with the previous month, a lag of 2 means you are comparing each month with the month before the previous one, and so on.

2. **Correlation**: The correlation between two variables measures how closely they are related. In the context of the ACF, the correlation is between the time series and its lagged versions.

3. **Autocorrelation Coefficient**: The ACF at lag $k$ is the correlation coefficient between the time series and its lagged version at lag $k$. It is denoted as $\rho_k$.

#### Mathematical Definition:

The autocorrelation function at lag $k$ is given by:

$$
\rho_k = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}
$$

where:
- $X_t$ is the value of the time series at time $t$.
- $X_{t-k}$ is the value of the time series at time $t-k$ (i.e., $k$ time periods earlier).
- $\text{Cov}(X_t, X_{t-k})$ is the covariance between $X_t$ and $X_{t-k}$.
- $\text{Var}(X_t)$ is the variance of the time series.

#### Interpretation:

- **Lag 0 (k=0)**: The autocorrelation at lag 0 is always 1 because it represents the correlation of the time series with itself, which is perfect.
- **Positive Autocorrelation**: If $\rho_k > 0$, it indicates that the time series values at lag $k$ are positively correlated. This means that high values in the time series tend to be followed by high values, and low values tend to be followed by low values.
- **Negative Autocorrelation**: If $\rho_k < 0$, it indicates that the time series values at lag $k$ are negatively correlated. This means that high values tend to be followed by low values, and vice versa.
- **Zero Autocorrelation**: If $\rho_k \approx 0$, it indicates that there is no significant linear relationship between the time series and its lagged version at lag $k$.

#### Example:

Consider a simple time series representing the monthly sales of a product over a year:

$$
\text{Sales} = \{100, 120, 110, 130, 140, 150, 160, 170, 180, 190, 200, 210\}
$$

To calculate the ACF at lag 1, we compare each month's sales with the sales from the previous month:

$$
\text{Lag 1: } \{120, 110, 130, 140, 150, 160, 170, 180, 190, 200, 210\}
$$

We then calculate the correlation coefficient between the original series and the lagged series. If the correlation coefficient is positive and close to 1, it indicates a strong positive autocorrelation at lag 1, meaning that sales in one month are highly correlated with sales in the previous month.

#### Visual Representation:

The ACF is often visualized using a plot called the **Autocorrelation Plot** or **ACF Plot**. This plot shows the autocorrelation coefficients for different lags on the y-axis and the lags on the x-axis. The plot typically includes a dashed line representing the significance level, which helps in determining whether the autocorrelation coefficients are statistically significant.

### Conclusion:

The Autocorrelation Function is a powerful tool in time series analysis that helps in understanding the internal structure of the data, identifying patterns, and selecting appropriate models. By examining the ACF, analysts can determine whether a time series is stationary, has a trend, or exhibits seasonality, which are crucial steps in building accurate forecasting models.