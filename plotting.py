import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import argparse
import math

def calculate_periods(peaks, times):
        periods = []
        for i in range(1, len(peaks)):
            period = times[peaks[i]] - times[peaks[i - 1]]
            periods.append(period)
        
        try:
            average_period = sum(periods) / len(periods)
            return average_period
        except ZeroDivisionError:
            print("No period recorded, error plotting")
            return 0

def plot_angle(df, output_path="angle_graph.png"):

    """
    The driver function will read the csv or otherwise provide the angles and times arrays.
    """

    angles = df["Angle(deg)"].values
    times = df["Time(s)"].values

    # Find peaks (maxima) in the angle data
    peaks, _ = find_peaks(angles)

    # Calculate periods

    average_period = calculate_periods(peaks, times)

    # Plot Angle vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Time(s)"], df["Angle(deg)"], marker='o', linestyle='-', color='b')
    plt.title("Pendulum Angle vs Time")
    plt.suptitle(f"Average Period: {average_period:.4f} seconds")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.grid(True)
    plt.savefig(f"output/{output_path}", format="png")
    plt.show()

def fit_amplitude(df, output_path="amplitude_decay.png"):

    angles = df["Angle(deg)"]

    angles = angles.apply(lambda x: math.radians(x))

    times = df["Time(s)"]

    peaks, _ = find_peaks(angles)
    peak_times = times[peaks]
    peak_amplitudes = angles[peaks]

    error = 0.5 * np.ones_like(peak_times)
    error = np.radians(error)
    xerr = 0.5 * np.ones_like(peak_times)

    # Fit the peak heights to an exponential function A * exp(-gamma * t)
    def exponential_decay(t, A, gamma):
        return A * np.exp(-gamma * t)

    # Perform curve fitting to the peak data
    popt, _ = curve_fit(exponential_decay, peak_times, peak_amplitudes)

    # Fitted parameters
    fitted_A, fitted_gamma = popt

    # Generate the fitted curve
    fitted_peak_heights = exponential_decay(peak_times, fitted_A, fitted_gamma)
    
    residual = peak_amplitudes - fitted_peak_heights

    # Find tau & Q factor
    tau = 1 / fitted_gamma # Calculations see README.md
    average_period = calculate_periods(peaks, times)
    q_factor = math.pi * tau / average_period

    plt.figure(figsize=(10, 6))  # Decrease figure height for shorter plots

    plt.subplot(2, 1, 1)
    plt.plot(times, angles, label='Damped Oscillator', color='blue')
    plt.errorbar(peak_times, peak_amplitudes, fmt='ro', yerr=error, label='Peaks')
    plt.plot(peak_times, fitted_peak_heights, 'g--', label=f'Fitted: {fitted_A:.3f}*exp(-{fitted_gamma:.3f}*t)')
    plt.suptitle(f"Tau: {tau:.3f} seconds, T: {average_period:.3f}, Q factor: {q_factor:.3f}")
    plt.title("Amplitude vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (radians)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    # Plot the residuals
    plt.errorbar(peak_times, residual, yerr=error, fmt='o', label='Residuals')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Residuals of the Exponential Fit')
    plt.legend()

    plt.tight_layout(pad=2.0)  # Add padding between plots
    plt.savefig(f"output/{output_path}", format="png")
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Plot pendulum angle from CSV data.")
    parser.add_argument("--path", type=str, default="positions.csv", help="Path to the CSV file containing the data.")
    args = parser.parse_args()

    # If called from main you must specify a CSV file path
    df = pd.read_csv(args.path, header=0)
    
    fit_amplitude(df)
    # plot_angle(df)
