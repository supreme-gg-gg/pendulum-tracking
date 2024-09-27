import cv2
from plotting import fit_amplitude
from detection import detect_object
import sys, pandas as pd, math, argparse

def live_track(tracker, video_src):

    roi = detect_object()

    cap = cv2.VideoCapture(int(video_src))

    # Exit if video not opened.
    if not cap.isOpened():
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

    ret, frame = cap.read()
    if not ret:
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

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, roi)

    position_log = []

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

    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:

        ok, frame = cap.read()
        if not ok:
            break

        ok, bbox = tracker.update(frame)

        # Calculate the real-life timestamp of this frame
        current_time = (current_frame / fps)
        
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
                position_log.append([round(current_time, 3), adj_x, adj_y, round(angle, 3)])\

            cv2.putText(frame, "Tracking failure detected", (100,140), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2)
            # Display information on screen
            cv2.putText(frame, f"Time: {current_time:.2f}, X: {adj_x:.2f}, Y:{adj_y:.2f}", (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,170,50),2)
            cv2.putText(frame, f"Angle:{angle:.2f}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,170,50),2)
            
            # Display result
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tracking", min(frame_width, 800), min(frame_height, 600))  # Limit window size
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

        current_frame += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(position_log, columns=["Time(s)", "X", "Y", "Angle(deg)"])
    df.to_csv("live_track_output.csv", index=False)
    print(f"Position data saved.")

    # plot_angle(df, output_path=png_path)
    fit_amplitude(df, output_path="live_track_output.png")
    # print(f"Plot saved to {png_path}")
    print(f"Amplitude plot saved.") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify tracker and video source.")

    parser.add_argument(
        '--tracker',
        type=str,
        help='Please use capitalized name, see README.md',
        default="KCF"
    )

    parser.add_argument(
        '--source',
        type=str,
        default="0",
        help='Video source, webcam is usually 0'
    )

    args = parser.parse_args()

    trackers = {
        "CSRT": cv2.legacy.TrackerCSRT_create,
		"KCF": cv2.legacy.TrackerKCF_create,
		"BOOSTING": cv2.legacy.TrackerBoosting_create,
		"MIL": cv2.legacy.TrackerMIL_create,
		"TLD": cv2.legacy.TrackerTLD_create,
		"MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
		"MOSSE": cv2.legacy.TrackerMOSSE_create
    }

    tracker = trackers[args.tracker]()

    print(f"Using tracker: {args.tracker}")
    print(f"Using video source: {args.source}")

    live_track(tracker, args.source)
