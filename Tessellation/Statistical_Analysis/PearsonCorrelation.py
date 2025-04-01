import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress

# Example data (replace with your actual data)
volume_ischemic = np.array([27.899, 13.702, 13.38345, 4.2301, 0.88032, 3.92015, 10.7587, 29.35993, 6.17318, 24.20351, 8.2792, 6.055, 0.92835, 9.24592, 11.78987, 4.90961, 1.27014, 12.57777, 35.49333, 18.48826, 1.60126, 13.87459, 15.23712, 3.54087, 16.88507, 14.75311, 10.77631, 16.93876, 6.77203, 4.9408, 4.70413, 3.65884, 2.57141])
volume_territory = np.array([32.257, 26.903, 17.49709, 14.48012, 32.86884, 23.45954, 15.35299, 69.13086, 22.33032, 23.26416, 29.80469, 23.05939, 23.56842, 18.30134, 20.68575, 23.17049, 32.9959, 17.87324, 29.61121, 39.2276, 7.54242, 30.59204, 57.24639, 16.04462, 42.85614, 26.75049, 14.34265, 31.56433, 12.70355, 19.68597, 35.0122, 8.85681, 13.88978])

volume_territory_5 = np.array([32.257, 26.903, 17.49709, 14.48012, 15.35299, 69.13086, 22.33032, 23.26416, 29.80469, 23.05939, 18.30134, 20.68575, 23.17049, 17.87324, 29.61121, 39.2276, 30.59204, 57.24639, 42.85614, 26.75049, 14.34265, 31.56433, 12.70355, 35.0122])
volume_ischemic_5  = np.array([27.899, 13.702, 13.38345, 4.2301, 10.7587, 29.35993, 6.17318, 24.20351, 8.2792, 6.055, 9.24592, 11.78987, 4.90961, 12.57777, 35.49333, 18.48826, 13.87459, 15.23712, 16.88507, 14.75311, 10.77631, 16.93876, 6.77203, 4.70413])

volume_territory_10 = np.array([32.257, 26.903, 17.49709, 15.35299, 69.13086, 23.26416, 29.80469, 18.30134, 20.68575, 17.87324, 29.61121, 39.2276, 30.59204, 57.24639, 42.85614, 26.75049, 14.34265, 31.56433])
volume_ischemic_10  = np.array([27.899, 13.702, 13.38345, 10.7587, 29.35993, 24.20351, 8.2792, 9.24592, 11.78987, 12.57777, 35.49333, 18.48826, 13.87459, 15.23712, 16.88507, 14.75311, 10.77631, 16.93876])


# Calculate Pearson correlation coefficient
r, p_value = pearsonr(volume_ischemic, volume_territory)
r5, p_value5 = pearsonr(volume_ischemic_5, volume_territory_5)
r10, p_value10 = pearsonr(volume_ischemic_10, volume_territory_10)

print("all data:", len(volume_ischemic), r, p_value)
print("5%:", len(volume_ischemic_5), r5, p_value5)
print("10%:", len(volume_ischemic_10), r10, p_value10)


MBF_ischemic_5 = np.array([0.47587, 0.55952, 0.56239, 0.61679, 0.5155, 0.58192, 0.58808, 0.43058, 0.72135, 0.55773, 0.54798, 0.50697, 0.51009, 0.38941, 0.52149, 0.70745, 0.60783, 0.50648, 0.5173, 0.56167, 0.54521, 0.47597, 0.48392, 0.504])
MBF_territory_5 = np.array([0.57323, 0.69927, 0.66714, 0.80698, 0.61942, 0.75818, 0.87044, 0.52901, 0.85057, 0.82458, 0.72113, 0.66772, 0.85138, 0.55805, 0.65973, 0.80379, 0.797, 0.77392, 0.74476, 0.68491, 0.68475, 0.66164, 0.64493, 0.79937])

r_MBF, p_value_MBF = pearsonr(MBF_ischemic_5, MBF_territory_5)

print("MBF:", r_MBF, p_value_MBF)

'''
# Linear regression for the trend line
slope, intercept, _, _, _ = linregress(volume_ischemic, volume_territory)
x_fit = np.linspace(min(volume_ischemic), max(volume_ischemic), 100)
y_fit = slope * x_fit + intercept

# Plot scatter plot
plt.figure(figsize=(6,6))
sns.scatterplot(x=volume_ischemic, y=volume_territory, color='blue', label="Data points")

# Plot trend line
plt.plot(x_fit, y_fit, 'r--', label=f"Trend line (r={r:.2f})")

# Plot x = y line
plt.plot(x_fit, x_fit, 'k-', label="x = y line")

# Labels and title
plt.xlabel("Volume_Ischemic (mL)")
plt.ylabel("Volume_Territory (mL)")
plt.title("Scatter Plot with Pearson Correlation")
plt.legend()
plt.grid()

# Show plot
plt.show()

# Print Pearson correlation
print(f"Pearson correlation coefficient: r = {r:.2f}, p-value = {p_value:.4f}")
'''