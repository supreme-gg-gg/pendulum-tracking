# PHY180 -- Pendulum Motion Tracker

This project utilizes OpenCV's Trackers API to develop an automated tracking system for processing recorded images of pendulum motion. You should put all the video you wish to process in one directory and the program will automatically process all the files.

**An angle-time graph will be produced with average period, while logging X, Y, and angle to a csv file.** For each video there will be one csv file and one graph in the `ouput/` directory.

## Requirements

OpenCV is originally a C computer vision library written in C++, but its Python API is much more convenient in development. You need to install the package in order to run this script.

```
pip install opencv-python
conda install opencv
```

**For your convenience, you can initialize a new virtual environment using the requirement files we generated.** If you don't know how to use either Google or ChatGPT can help.

```
pip install -r requirements.txt
conda env create -f environment.yaml
```

## Usage

The following script will process all the files in the specified directory:

```
python3 tracker.py --tracker <tracker_type> --source <path-to-videos>
```

The tracker is defaulted to `KCF` (Kernalized Correlation Filter). All available options for OpenCV 4.0+ are: `tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']`.

The user will be prompted to:

- Use your mouse to click on the first frame to select the origin to adjust x-y coordinates

- Draw a bounding box on the region of interest (ROI) (i.e. the object you want to track)

- Click `Esc` if you want to quit the video midway, otherwise it will exit when finished

- The next video will begin automatically and the process repeats until all videos are processed

Calling main tracker script will automatically generate plots in the output directory. If you want to plot a specific CSV file, you can do that with `python3 plotting.py`. Since you will rarely be calling this manually, argparse wasn't implemented so just modify the file path using a text editor.

## Known Issues

### OpenCV Version

OpenCV has a very bad API maintenace and legacy support. Once a new version is released, certain API calls are changed / deprecated. This is why if you are using an older version the script might not work, so please search up solutions on StackOverflow or ChatGPT/Copilot. A common error looks like:

```
Traceback (most recent call last):
  File "/Users/xxx/xxx/xxx/opencv-tracking/tracker.py", line 174, in <module>
    "TLD": cv2.TrackerTLD.create,
           ^^^^^^^^^^^^^^
AttributeError: module 'cv2' has no attribute 'TrackerTLD'
```

### Tracker Error

It has come to our attention that for the first few seconds of certain video tracker failed to work properly. This caused missing data points at both ends. The best solution would be to truncate the inconsistent datapoints. **This will be implemented in future patches.**

### Failing to Track

If you have a video with too high frame rate (e.g. 240fps), it is likely that the tracker can fail. Our performance benchmark was tested on 30-60 fps, you should investigate the issue yourself or try to reduce the frame rate as a precaution.

## TODO

1. In future editions, we have plan to implement automatic object identification either through object detection OR inferencing from manual selection only for the first video.

2. Process the dataframe: remove inconsistency on both ends and filter / double-check the period for anomalies.
