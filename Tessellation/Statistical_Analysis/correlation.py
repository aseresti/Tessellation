import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress, wilcoxon

def ScatterPlot(x, y, xlabel, ylabel, r, p):
    
    slope, intercept, _, _, _ = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept


    plt.figure(figsize=(6,6))
    sns.scatterplot(x=x, y=y, color='red')#, label="Data points")

    plt.plot(x_fit, y_fit, 'r--', label=f"Trend line (r={r:.2f})")

    plt.plot(x_fit, x_fit, 'k-', label="x = y line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    #plt.legend()
    #plt.grid()
    plt.text(0.02, 0.95, f"p = {p:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.02, 0.88, f"r = {r:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')



if __name__ == "__main__":

    velocity_inv = np.array([8.02, 9.19, 7.39, 8.995, 5.635, 5.32, 7.46, 9.945, 10.55, 7.86, 6.26, 6.73, 13.275, 9.95, 10.12, 7.365, 7.075, 8.09])
    velocity_cfd = np.array([6.23, 6.95, 17.88, 7.07, 10.00, 6.31, 16.84, 9.42, 4.50, 8.65, 4.74, 4.49, 8.59, 6.28, 5.11, 4.39, 11.40, 11.91])

    flow_inv = np.array([64.21, 10.46, 36.91, 34.92, 39.02, 11.93, 40.10, 87.94, 32.34, 96.29, 13.41, 32.55, 52.80, 18.05, 41.00, 16.93, 12.10, 15.44])
    flow_cfd = np.array([49.86, 7.91, 89.29, 27.46, 69.26, 14.16, 90.51, 83.33, 13.80, 106.00, 10.16, 21.71, 34.18, 11.39, 20.70, 10.09, 19.50, 3.64])

    pressure_inv = np.array([115.0, 119.0, 81.0, 71.0, 102.0, 90.0, 105.0, 110.0, 116.0, 90.0, 61.0, 82.0, 63.0, 66.0, 83.0, 85.0, 98.0, 81.0])
    pressure_cfd = np.array([109.4, 109.2, 78.6, 76.7, 102.7, 90.6, 112.6, 112.7, 112.5, 97.3, 74.2, 90.7, 81.4, 79.1, 74.5, 74.3, 92.7, 94.4])


    # Calculate Pearson correlation coefficient
    rv, p_value_v = pearsonr(velocity_cfd, velocity_inv)
    rf, p_value_f = pearsonr(flow_cfd, flow_inv)
    rp, p_value_p = pearsonr(pressure_cfd, pressure_inv)

    ScatterPlot(velocity_inv, velocity_cfd, r"$Velocity_{Doppler}$ (cm/s)", r"$Velocity_{CFD}$ (cm/s)", rv, p_value_v)
    ScatterPlot(flow_inv, flow_cfd, r"$Q_{Doppler}$ (mL/min)", r"$Q_{CFD}$ (mL/min)", rf, p_value_f)
    ScatterPlot(pressure_inv, pressure_cfd, r"$Pressure_{Invasive}$ (mmHg)", r"$Pressure_{CFD}$ (mmHg)", rp, p_value_p)


    #Wilcoxon Signed Rank Test
    res_v = wilcoxon(velocity_inv, velocity_cfd)
    res_f = wilcoxon(flow_inv, flow_cfd)
    res_p = wilcoxon(pressure_inv, pressure_cfd)

    print("velcoity wilcoxon test:", res_v.statistic, res_v.pvalue)
    print("flow wilcoxon test:", res_f.statistic, res_f.pvalue)
    print("pressure wilcoxon test:", res_p.statistic, res_p.pvalue)