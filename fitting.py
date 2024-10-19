import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Define the functions
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def power_func(x, a, b):
    return a * x**b

# Ask user to upload a CSV file
Tk().withdraw()  # Close the root window from tkinter
print("Please select a CSV file containing 'x', 'y', and 'uncertainty' columns.")
file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected. Exiting...")
    exit()

# Load the CSV file, assuming the first row contains headers
try:
    data = pd.read_csv(file_path)  # pandas automatically reads the first row as headers
    x_data = data.iloc[:, 0].values  # First column for x values
    y_data = data.iloc[:, 1].values  # Second column for y values
    y_err = data.iloc[:, 2].values  # Third column for uncertainty (error bars)
    x_label = data.columns[0]  # Use the first header for the x-axis label
    y_label = data.columns[1]  # Use the second header for the y-axis label
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Ask user which function to fit
print("Choose the function to fit:")
print("1. Linear")
print("2. Quadratic")
print("3. Power")
choice = input("Enter the number of your choice: ")

# Select the appropriate function
if choice == '1':
    func = linear
    p0 = [1, 1]  # Initial guess for linear parameters
    func_name = "Linear"
elif choice == '2':
    func = quadratic
    p0 = [1, 1, 1]  # Initial guess for quadratic parameters
    func_name = "Quadratic"
elif choice == '3':
    func = power_func
    p0 = [1, 1]  # Initial guess for power function parameters
    func_name = "Power"
else:
    print("Invalid choice")
    exit()

# Fit the function to the data with uncertainties
if np.any(y_err == 0):
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0)
else:
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0, sigma=y_err, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))  # Uncertainties in parameters

# Print the results
if func_name == "Linear":
    print(f"Fitted equation: y = {popt[0]:.3f}x + {popt[1]:.3f}")
    print(f"Uncertainty in m: {perr[0]:.3f}")
    print(f"Uncertainty in b: {perr[1]:.3f}")
elif func_name == "Quadratic":
    print(f"Fitted equation: y = {popt[0]:.3f}x^2 + {popt[1]:.3f}x + {popt[2]:.3f}")
    print(f"Uncertainty in a (x^2 term): {perr[0]:.3f}")
    print(f"Uncertainty in b (x term): {perr[1]:.3f}")
    print(f"Uncertainty in c (constant): {perr[2]:.3f}")
elif func_name == "Power":
    print(f"Fitted equation: y = {popt[0]:.3f}x^{popt[1]:.3f}")
    print(f"Uncertainty in a: {perr[0]:.3f}")
    print(f"Uncertainty in b: {perr[1]:.3f}")

# Plot the data with error bars and the fit
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label="Data", color='red', ecolor='black', capsize=5)
plt.plot(x_data, func(x_data, *popt), label=f"{func_name} fit", color='blue')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f"{func_name} Fit")
plt.legend()
plt.show()