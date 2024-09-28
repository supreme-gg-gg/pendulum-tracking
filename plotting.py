import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import argparse
import math

def binning(trials, max_time, bin_size=1):
    """
    Bins the amplitude data from multiple trials into specified time bins and calculates the mean and uncertainty of the mean for each bin.
    Args:
        trials (list of list of tuples): A list where each element is a trial, and each trial is a list of (timestamp, amplitude) tuples.
        max_time (float): The maximum time up to which the data should be binned.
        bin_size (float, optional): The size of each time bin. Default is 1.
    Returns:
        tuple: A tuple containing:
            - bin_centers (numpy.ndarray): The centers of the time bins.
            - mean_amplitudes (numpy.ndarray): The mean amplitude for each time bin.
            - uncertainty_mean (list): The uncertainty of the mean amplitude for each time bin.
    """

    bins = np.arange(0, max_time + bin_size, bin_size)

    # Initialize lists to hold max amplitudes per trial
    max_amplitudes_per_trial = []

    # Find max amplitude for each trial
    for trial in trials:
        max_amplitude = []
        for timestamp, amplitude in trial:
            max_amplitude.append(amplitude)
        max_amplitudes_per_trial.append(max(max_amplitude))

    # Now bin the max amplitudes
    mean_amplitudes = []
    std_amplitudes = []

    for start in bins[:-1]:
        bin_amplitudes = []
        for trial in trials:
            for timestamp, amplitude in trial:
                if start <= timestamp < start + bin_size:
                    bin_amplitudes.append(amplitude)

        if bin_amplitudes:
            mean_amplitudes.append(np.mean(bin_amplitudes))
            std_amplitudes.append(np.std(bin_amplitudes))
        else:
            mean_amplitudes.append(np.nan)
            std_amplitudes.append(np.nan)

    # Convert to numpy arrays for further use or plotting
    mean_amplitudes = np.array(mean_amplitudes)
    std_amplitudes = np.array(std_amplitudes)
    
    # Calculate uncertainty of the mean
    uncertainty_mean = [std / np.sqrt(len(trials)) if len(trials) > 0 else np.nan for std in std_amplitudes]
    
    # Create a mask that identifies positions where both arrays are not NaN
    valid_mask = ~np.isnan(mean_amplitudes) & ~np.isnan(uncertainty_mean)

    # Apply the mask to filter out the NaN values from both arrays
    mean_amplitudes = mean_amplitudes[valid_mask]
    uncertainty_mean = std_amplitudes[valid_mask]

    # Calculate bin centers for plotting
    bin_centers = bins[:-1] + bin_size / 2
    bin_centers = bin_centers[valid_mask]

    return bin_centers, mean_amplitudes, uncertainty_mean

def calculate_periods(peaks, times):
        """
        Calculate the average period between peaks in a time series.
        Args:
            peaks (list of int): Indices of the peaks in the time series.
            times (list of float): Time values corresponding to each index in the time series.
        Returns:
            float: The average period between consecutive peaks. Returns 0 if no periods are recorded.
        """

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
    Plots the angle of a pendulum over time and saves the plot as an image.
    Args:
        df (pandas.DataFrame): DataFrame containing the pendulum data with columns "Angle(deg)" and "Time(s)".
        output_path (str): The file path where the plot image will be saved. Default is "angle_graph.png".
    Returns:
        None
    The function performs the following steps:
    1. Extracts the angle and time data from the DataFrame.
    2. Finds the peaks (maxima) in the angle data.
    3. Calculates the average period of the pendulum.
    4. Plots the angle vs. time graph.
    5. Adds a title, labels, and grid to the plot.
    6. Saves the plot as a PNG image to the specified output path.
    7. Displays the plot.
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

def fit_amplitude(df, ret=False, err=None, output_path="amplitude_decay.png"):
    """
    Fits the amplitude of a damped oscillator to an exponential decay function and plots the results.
    Parameters:
    df (pandas.DataFrame): DataFrame containing the columns "Angle(deg)" and "Time(s)".
    ret (bool, optional): If True, returns the amplitude over time data. Default is False.
    err (numpy.ndarray, optional): Error bars for the peak amplitudes. If None, default error bars are used. Default is None.
    output_path (str, optional): Path to save the output plot. Default is "amplitude_decay.png".
    Returns:
    list of tuples: If ret is True, returns a list of tuples containing peak times and peak amplitudes.
    Notes:
    - The function converts angles from degrees to radians.
    - Peaks in the amplitude data are identified and fitted to an exponential decay function A * exp(-gamma * t).
    - The fitted parameters (A and gamma) are used to calculate the time constant (tau) and the quality factor (Q).
    - The function generates a plot with the original data, identified peaks, fitted curve, and residuals.
    - The plot is saved to the specified output path.
    """

    angles = df["Angle(deg)"]

    angles = angles.apply(lambda x: math.radians(x))

    times = df["Time(s)"]

    peaks, _ = find_peaks(angles)
    peak_times = times[peaks]
    peak_amplitudes = angles[peaks]
    
    error = np.radians(0.5 * np.ones_like(peak_times)) if err == None else err
    # xerr = 0.5 * np.ones_like(peak_times)

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
    plt.savefig(f"{output_path}", format="png")
    plt.show()

    if ret:
        return list(zip(peak_times, peak_amplitudes)), q_factor
    
def amplitude_decay(times, peaks, error, q_factor, output_path="decay_fit.png"):
    
    """
    Plots the amplitude decay of a signal over time and fits an exponential decay curve to the data.
    Parameters:
    times (array-like): Array of time values.
    peaks (array-like): Array of peak amplitude values corresponding to the time values.
    error (array-like): Array of error values for the peak amplitudes.
    output_path (str, optional): Path to save the output plot image. Default is "amplitude_decay_trials.png".
    Returns:
    None
    The function performs the following steps:
    1. Defines an exponential decay function.
    2. Fits the exponential decay function to the provided peak data.
    3. Calculates the residuals between the actual peaks and the fitted curve.
    4. Plots the original peak data with error bars and the fitted exponential decay curve.
    5. Plots the residuals of the fit.
    6. Saves the plot to the specified output path and displays it.
    """

    def exponential_decay(t, A, gamma):
        return A * np.exp(-gamma * t)

    # Perform curve fitting to the peak data
    popt, _ = curve_fit(exponential_decay, times, peaks)

    # Fitted parameters
    fitted_A, fitted_gamma = popt

    # Generate the fitted curve
    fitted_peak_heights = exponential_decay(times, fitted_A, fitted_gamma)
    
    residual = peaks - fitted_peak_heights
    
    plt.figure(figsize=(10, 6))  # Decrease figure height for shorter plots

    plt.subplot(2, 1, 1)
    plt.errorbar(times, peaks, fmt='ro', yerr=error, label='Peaks')
    plt.plot(times, fitted_peak_heights, 'g--', label=f'Fitted: {fitted_A:.3f}*exp(-{fitted_gamma:.3f}*t)')
    plt.suptitle(f"Average Q factor: {q_factor:.3f}")
    plt.title("Amplitude vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (radians)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    # Plot the residuals
    plt.errorbar(times, residual, yerr=error, fmt='o', label='Residuals')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Residuals of the Exponential Fit')
    plt.legend()

    plt.tight_layout(pad=2.0)  # Add padding between plots
    plt.savefig(f"{output_path}", format="png")
    plt.show() 

    print(f"Amplitude decay fit plot saved to {output_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Plot pendulum angle from CSV data.")
    parser.add_argument("--path", type=str, default="positions.csv", help="Path to the CSV file containing the data.")
    args = parser.parse_args()

    # If called from main you must specify a CSV file path
    df = pd.read_csv(args.path, header=0)
    
    # USAGE: COMMENT OR UNCOMMENT THE FOLLOWING LINES BEFORE RUNNING THIS FILE

    fit_amplitude(df)
    # plot_angle(df)
