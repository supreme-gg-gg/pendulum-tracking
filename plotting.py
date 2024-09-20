import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_angle(from_df=None, csv_src="positions.csv", output_path="angle_time.png"):

    """
    By default we assume this function is used to plot a graph by reading from an existing csv. 
    When called from tracker automatically, it will use the df provided.
    """

    df = pd.read_csv(csv_src, header=0) if from_df is None else from_df

    # Find peaks (for period calculation)
    angles = df["Angle(deg)"].values
    times = df["Time(s)"].values

    # Find peaks (maxima) in the angle data
    peaks, _ = find_peaks(angles)

    # Calculate periods
    def calculate_periods(peaks, times):
        periods = []
        for i in range(1, len(peaks)):
            period = times[peaks[i]] - times[peaks[i - 1]]
            periods.append(period)
        return periods

    periods = calculate_periods(peaks, times)
    # print(periods) # TODO: Some periods are not correct judging from the output...
    try:
        average_period = sum(periods) / len(periods)
    except ZeroDivisionError:
        print("No period recorded, error plotting")
        return

    # Plot Angle vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Time(s)"], df["Angle(deg)"], marker='o', linestyle='-', color='b')
    plt.title("Pendulum Angle vs Time")
    plt.suptitle(f"Average Period: {average_period:.4f} seconds")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.grid(True)
    plt.savefig(f"{output_path}", format="png")
    plt.show()

if __name__ == "__main__":
    plot_angle()