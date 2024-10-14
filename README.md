# Automated Tracking & Analysis For Pendulum Motion

This project utilizes OpenCV's Trackers API to develop an automated tracking system for processing recorded videos of pendulum motion. You should put all the video you wish to process in one directory and the program will automatically process all the files.

> The code is open source and feel free to fork and use it for your project. However, you must credit the creator if the code was used as a part of any academic assignment. Consider giving me a star for the repo too XD

**An angle-time graph will be produced with average period, while logging X, Y, and angle to a csv file. For an entire list of supported graphs, please see [the graphs section](#supported-graphs)** For each video there will be one csv file and one or more graph in the `ouput/` directory.

![Sample graph](sample/angle-time.png)

![Sample for amplitude decay](sample/amplitude_2.png)

## Table of Contents

- [Input Requirements](#input-requirements)
- [Contact Me and Contributions](#contact-me-and-contributions)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Main Tracker](#main-tracker)
  - [Multiple Trials](#multiple-trials)
  - [Supported Graphs](#supported-graphs)
  - [Live Tracking](#live-tracking)
- [Known Issues](#known-issues)
  - [OpenCV Version](#opencv-version)
  - [Tracking Instability](#tracking-instability)
  - [Inaccurate Period](#inaccurate-period)
  - [Difficulty in Picking Origin](#difficulty-in-picking-origin)
  - [Too few oscillations](#too-few-oscillations)
- [Citation](#citation)

## Background

Often physics projects require low-budget and efficient solutions. This is a great example. In short -- simple code, great performane, and best of all: **intuitive usage**. Online autotrackers are either unreliable or difficult to use. UX was obviously not a concern to those developers. Otherwise, these alternatives are packed with unnecessary features that only create confusion. The story is different here, as we embrace simplicity while getting the job done -- and doing it well.

[Get started with the all-in-one beginner-friendly solution to tracking.](#requirements)

## Input Requirements

The program works best with videos recorded at _**60fps**_. Most modern smartphones support this framerate and you can check it manually yourself. Technically, it would work for all other framerates (30 or 240) but this could cause unexpected errors with tracker or a wrong time scale.

> NOTE: You can expect bug fixes in future patches. Please create a GitHub Issue if you run into difficulties.

## Contact Me and Contributions

If you wish to contribute or have questions, reach out to Jet Chiang at [my email](mailto:jetjiang.ez@gmail.com). The project might not be actively maintained over the year but I am happy to help if possible.

We welcome any new contributions regardless of your skill level. This can be a great experience to practice python skills! However, before you create any PR with your commits **please test your code locally using video/csv sources**. This will make the code review process smoother!

## Requirements

OpenCV is open source computer vision library written in C++, but provides support for other languages such as Python via its API. You need to install the package from PyPI or Anaconda in order to run this script.

```
pip install opencv-python
conda install opencv
```

**For your convenience, you can initialize a new virtual environment using the requirement files we generated.** If you don't know how to use either Google it or ask ChatGPT help.

```
pip install -r requirements.txt
conda env create -f environment.yaml
```

## Usage

### Main Tracker

The following script will process all the files in the specified directory:

```
python3 tracker.py --tracker <tracker_type> --source <path-to-videos> --multi-trials <True/False>
```

The tracker is defaulted to `KCF` (Kernalized Correlation Filter). All available options for OpenCV 4.0+ are: `tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']`.

> By default we assume you want to run each video file independently. If you want to consider them as different samples, you should set `--multi-trials True` or using the flag `-m True`. For details, see [the subsequent section](#multiple-trials)

The user will be prompted to:

- Use your mouse to click on the first frame to select the origin to adjust x-y coordinates

- Draw a bounding box on the region of interest (ROI) (i.e. the object you want to track)

- Click `Esc` if you want to quit the video midway, all data and plots will still be recorded

- The next video will begin automatically and the process repeats until all videos are processed

**If you wish to output a different type of graph supported in the list below, feel free to modify part of the program. However, you are discouraged to do it unless you are completely sure what you are doing.**

### Multiple Trials

It is common in experiments to conduct multiple trials on the same initial conditions in order to assess type A uncertainty, or **random error**. To facilitate this, you can set the corresponding command line argument `--multiple-trials True`. By doing so:

- The function will repeated run over all files in the directory and generate individual graphs

- Also store the amplitude-time data from each file and perform binning to aggregate data

- Calculate the mean and uncertainty of trials combined and generate a plot with curve fitting and errors

### Supported Graphs

The following graphs or outputs are supported by the graph as per the current release:

1. Regular angle vs. time graph with period computed

2. Amplitude vs. time graph with exponential function fit and Q value computed

3. Exponential decay of amplitude with random error uncertainty and mean using multiple trials

Calling main tracker script will automatically generate one or more plots in the output directory, see [tracker intro](#multiple_trials). If you wish to plot from a csv direclty without analysing the video, the module is also offered as a standalone graphing tool:

```
python3 plotting.py --csv_files_path <file-path> --csv_files_directory <dir-path>
```

Note the following:

- You must provide either a single file or a directory of csv files for the program to loop over

- Plots will be generated in `output/` as well, **so be very cautious of the possibility of overwriting files!**

- The folder you provided **MUST NOT** contain anything other than csv files. Therefore, we do not recommend providing the `output/` directory as a source but rather you should move all csv you want to plot into a separate directory

> By default, we invoke both `plot_angle()` and `fit_amplitude()` when the plotting module is called directly. Feel free to locally modify the plotting script if you only wish to see one type of graph.

### Live Tracking

The live tracking functionality is built on top of two methods implemented using OpenCV:

1. Tracking (`tracker.py`): This is a standalone tracking module that was originally developed using a series of OpenCV Python API trackers. See [the section above](#main-tracker) for more information.

2. Real-time object detection (`detection.py`): A new script intending to improve usability and extend support to live tracking (i.e. using camera instead of pre-recorded video).

**OD is required because before we call the tracking module we need to know where is the initial position of the object.** Fine tuning OD models like YOLO, despite considerably better performance, would be time-consuming and therefore we incline towards simpler efficient (yet less accurate) algorithms such as colour tracking. Once we obtained the initial bounding box (ROI) of the object and the use selected `Enter`, we will automatically call the tracker module to do its regular job.

### Finding Q factor

Please google / ChatGPT if you don't know what is the Q factor for a pendulum motion, as we shall not spend time to discuss it here. The graphing module supports calculating Q factor automatically using the exponential decay model. **We calculate it both ways using tau and counting oscillations.** For the latter, we find where $\theta = \theta_0 \exp{-\pi/4}$ to estimate $\frac{Q}4$. This helps reduce time duration of the recording but feel free to increase it as needed. The results are displayed on a text box in the upper right corner of the second (residual) graph.

> WARNING: If your video is too short, the function will not raise an error but Q factor will be dispalyed as -1.

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

### Tracking Instability

In theory, the tracker will perform well even without a uniform background colour and with medium to high environemntal distraction and chaos. However, the most common form of tracking failure is losing the object during the first 10 seconds. Should this occur, the following countermeasures are recommended:

* Rerun the script with a different tracker, in most caes `CSRT` work well

* Skip that video and rerun the entire script again on that video itself, this usually solves all issues

* Select a tigter bounding box that only includes the box (i.e. does not include any part of your hand)

> Raise a GitHub Issue if severity is high. The issue is being investigated.

### Difficulty in Picking Origin

We expect a better GUI assitance for picking origin (including x and y axis) in the future. For now, estimate the origin and use a mouse to select the point. _Technically, this does not result in any implications in the result analysis_ since transformation of a function will not change its maximum and minimum, and in most cases we are only concerned with **periods** instead of specific angles.

### Too few oscillations

In the case of the error message below, it is likely that you stopped the video too early such that the script has not collected enough data points for curve fitting. The two solutions is either to avoid using `fit_amplitude()` or let the script run longer.

```
File "/Users/xxx/xxx/xxx/opencv-tracking/plotting.py", line 162, in fit_amplitude
  popt, _ = curve_fit(exponential_decay, peak_times, peak_amplitudes)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/Users/xxx/xxx/envs/opencv/lib/python3.12/site-packages/scipy/optimize/_minpack_py.py", line 1000, in curve_fit
  res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/Users/xxx/xxx/envs/opencv/lib/python3.12/site-packages/scipy/optimize/_minpack_py.py", line 424, in leastsq
  raise TypeError(f"Improper input: func input vector length N={n} must"
TypeError: Improper input: func input vector length N=2 must not exceed func output vector length M=1
```

## Citation

If you use this code for any academic purposes, please cite it as follows:

```
@misc{chiang2024opencvtracking,
  author = {Jet Chiang},
  title = {Automated Tracking And Analysis System for Pendulum Motion},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/supreme-gg-gg/opencv-tracking}},
}
```

Plagarising any part of the code without consent from the publisher is a serious academic offense.
