import cv2
import sys
import pandas as pd
import math
import argparse
import os

from plotting import plot_angle, fit_amplitude, binning, amplitude_decay

def process_video(tracker, video_path, csv_path, ret=False):
    """
    Process a video to track an object and log its position and angle over time.
    Args:
        tracker (cv2.Tracker): The OpenCV tracker object to use for tracking.
        video_path (str): Path to the input video file.
        csv_path (str): Path to save the CSV file containing the tracking results.
        ret (bool, optional): If True, returns a list of list of tuples containing the results. Defaults False.
    Returns:
        tuple: If ret is True, returns a tuple containing the fitted amplitude data and the total time of the video.
    Notes:
    - The user is required to manually select an equilibrium point and the Region of Interest (ROI) in the first frame.
    - The function logs the position and angle of the tracked object at regular intervals.
    - The results are saved to a CSV file, and an amplitude plot is generated and saved as a PNG file.
    - This function works closely with fit_amplitude and amplitude_decay
    """

    video = cv2.VideoCapture(video_path)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Before starting the tracking, let the user pick a equilibrium point as origin
    # The user should click on the frame to drop the point

    origin = None

    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal origin # update enclosing scope
            print(f"Equilibrium Point {origin} set at: X = {x}, Y = {y}")
            origin = (x, y)

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Limit window size, resize frame if necessary
    frame_height, frame_width = frame.shape[:2]
    cv2.namedWindow("Select Equilibrium Point", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Equilibrium Point", min(frame_width, 800), min(frame_height, 600))

    cv2.setMouseCallback("Select Equilibrium Point", select_point)

    while origin is None:
        # Display the frame and wait for user to click the equilibrium point
        cv2.imshow("Select Equilibrium Point", frame)

        # Use waitKey to allow OpenCV to process mouse events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit() 

    cv2.destroyWindow("Select Equilibrium Point")

    # Let the user select the ROI (bounding box, or Region of Interest) manually on the first frame
    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Object", min(frame_width, 800), min(frame_height, 600))  # Limit window size
    bounding_box = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Initialize tracker with first frame and bounding box
    global trackers
    cvtracker = trackers[tracker]()
    ok = cvtracker.init(frame, bounding_box)

    position_log = []
    # start_time = time.time()

    # Get the recorded fps (frames per second) from the video
    recorded_fps = video.get(cv2.CAP_PROP_FPS)

    # This is a temporary solution to a issue caused by arctan
    def normalize_angle(a):
        if a > 0:
            a -= 90  # Map (90°, 180°] to (-90°, 0°]
        elif a < 0:
            a += 90  # Map [-180°, -90°) to [0°, 90°)
        elif a == 90:
            a = 0
        return a
    
    adj_x, adj_y, angle = 0, 0, 0
    current_frame = 0

    while True:

        ok, frame = video.read()
        if not ok:
            break
            
        # Obtain frame per second (deprecated)
        '''
        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        '''

        ok, bbox = cvtracker.update(frame)

        # Calculate the real-life timestamp of this frame
        current_time = (current_frame / recorded_fps)
        
        if ok:

            x, y, w, h = map(int, bbox)

            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            center_x = x + w // 2
            center_y = y + h // 2

            # Adjust by the origin (equilibrium point)
            adj_x = center_x - origin[0]
            adj_y = origin[0] - center_y

            angle = math.degrees(math.atan(adj_y/adj_x)) if adj_x != 0 else 0
            # angle = math.atan(adj_y/adj_x) if adj_x != 0 else 0
            angle = normalize_angle(angle)
            
            if current_frame % 5 == 0:
                position_log.append([round(current_time, 3), adj_x, adj_y, round(angle, 3)])

            # Display information on screen
            cv2.putText(frame, f"Time: {current_time:.2f}, X: {adj_x:.2f}, Y:{adj_y:.2f}", (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,170,50),2)
            cv2.putText(frame, f"Angle:{angle:.2f}", (100,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,170,50),2)

        else:

            cv2.putText(frame, "Tracking failure detected", (100,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2)
        
        # Display result
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", min(frame_width, 800), min(frame_height, 600))  # Limit window size
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        current_frame += 1

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(position_log, columns=["Time(s)", "X", "Y", "Angle(deg)"])
    df.to_csv(csv_path, index=False)
    print(f"Position data saved to {csv_path}")

    png_path = os.path.splitext(csv_path)[0] + '.png'
    # plot_angle(df, output_path=png_path)
    if ret:
        data, q_factor = fit_amplitude(df, output_path=png_path, ret=True)
    else:
        fit_amplitude(df, output_path=png_path) 
    # print(f"Plot saved to {png_path}")
    print(f"Amplitude plot saved to {png_path}")

    if ret:
        return data, df["Time(s)"].iloc[-1], q_factor

def process_directory(tracker, directory, ret=False):
    """
    Processes all video files in a given directory using a specified tracker.
    Args:
        tracker: The tracking object or function used to process the videos.
        directory (str): The path to the directory containing video files.
        ret (bool, optional): If True, returns additional data for further processing. Defaults to False.
    Returns:
        None or tuple: If `ret` is True, returns a tuple containing:
            - times (list): List of time bins.
            - means (list): List of mean values for each bin.
            - uncertainties (list): List of uncertainties for each bin.
        Otherwise, returns None.
    Raises:
        FileNotFoundError: If the specified directory does not exist.
        Exception: If an error occurs during video processing.
    Notes:
        - Supported video file formats are: .mov, .MOV, .mp4, .avi, .mkv.
        - Processed video data is saved in the ./output directory with a corresponding CSV file.
    Example:
        process_directory(my_tracker, "/path/to/videos", ret=True)
    """

    # Get a list of all files in the directory
    video_files = [f for f in os.listdir(directory) if f.endswith(('.mov', '.MOV', '.mp4', '.avi', '.mkv'))]

    if ret:
        trials = []
        length = []
        q_facotrs = []

    if not video_files:
        print("No video files found in the directory.")
        return

    # Loop through each video file and process it
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)

        # By default all output are in ./output
        output_csv_path = os.path.join("./output", f"{os.path.splitext(video_file)[0]}_output.csv")

        # Call the video processing function
        if ret:
            data, time, q_factor = process_video(tracker, video_path, output_csv_path, ret)
            trials.append(data)
            length.append(time)
            q_facotrs.append(q_factor)
        else:
            process_video(tracker, video_path, output_csv_path)

    if ret:
        times, means, uncertainties = binning(trials, max_time=max(length))
        average_q = sum(q_facotrs) / len(q_facotrs) if len(q_facotrs) >= 0 else 0
        amplitude_decay(times, means, uncertainties, average_q, output_path=f"./output/decay_fit_{directory}.png")

    print(f"Processed {len(video_files)} video files in directory: {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify tracker and source directory.")

    parser.add_argument(
        '--tracker',
        type=str,
        help='Please use capitalized name, see README.md',
        default="KCF"
    )

    parser.add_argument(
        '--source',
        type=str,
        default="videos",
        help='Source directory path as string.'
    )

    parser.add_argument(
        '--multi-trials', '-m',
        type=str,
        default="False",
        help='Run multiple trials to get mean and error'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise NotADirectoryError(f"'{args.source}' is not a valid directory.")

    global trackers
    trackers = {
        "CSRT": cv2.legacy.TrackerCSRT_create,
		"KCF": cv2.legacy.TrackerKCF_create,
		"BOOSTING": cv2.legacy.TrackerBoosting_create,
		"MIL": cv2.legacy.TrackerMIL_create,
		"TLD": cv2.legacy.TrackerTLD_create,
		"MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
		"MOSSE": cv2.legacy.TrackerMOSSE_create
    }

    ret = True if args.multi_trials == "True" else False

    print(f"Using tracker: {args.tracker}")
    print(f"Operating on source directory: {args.source}")

    # CURRENTLY we manually set return as true or false!
    process_directory(args.tracker, args.source, ret)
