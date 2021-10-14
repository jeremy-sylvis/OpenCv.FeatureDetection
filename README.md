# OpenCv.FeatureDetection
A utility for image feature detection based on OpenCV (via EmguCV).

## Background
This project is intended to provide facilites for exploring OpenCV's feature detection (and, perhaps, other image recognition) capabilities against configured file inputs.

While going through the various OpenCV and EmguCV examples, I was displeased by the limited documentation for and sparse examples of effective use of OpenCV in various image recognition tasks. In lieu of comprehensive documentation, the best option available for determining suitable feature detection algorithms and configurations is to define testcases and evaluate performance against these testcases. To that end, this project was created. It serves the additional benefit of providing functional examples and notes of any insights learned along the way.

OpenCV (and, in its own ways, EmguCV) is hard to work with. With any luck, this project will make it *less* difficult for someone else.

## Usage
### Fuzzing feature detectors
OpenCV supports many feature detectors but has (at best) limited documentation. As a result, it's near-impossible to know if a given feature detector truly suits a usecase ahead-of-time.

This process exercises all available feature detectors (and many configurations) in OpenCV against a given set of example images and generates a report (and reference images) with the results. It is extremely computationally-intensive, but it will opt-in to use of OpenCL wherever OpenCV internally uses OpenCL.

It is expected that an 'input' folder is created containing all images expected to be used as testcases, along with a text file with an entry for every file describing:
* FileName to be processed
* Region of Interest (left, right, top, bottom coordinates) - this allows for determining an inlier/outlier (or, signal/noise) ratio

The generated report is CSV format for easy import to your spreadsheet of choice. It tracks:
* FileName processed
* Total feature count
* Feature count in defined Region of Interest
* Algorithm used
* Parameters for the algorithm used
* Execution time

Parameters required:
* `-InputPath '<path>'` - directory containing inputs
* `-OutputPath '<path>'` - directory intended to contain the output report and generated reference images

Execution:
`OpenCv.FeatureDetection.Console.exe -Operation FuzzFeatureDetectors -InputPath <myPath> -OutputPath <myOutputPath>`

## OpenCV Usage
### Mat vs UMat
Using UMat in favor of Mat comes with serious performance implications. Benchmark often to determine which is more appropriate for your scenario.

The primary benefit of UMat is that wherever possible (and, in theory, _sensible_) OpenCV will opt for an OpenCL-accelerated execution route. Understand that GPU-accelerated OpenCL contexts can be _incredibly_ expensive to set up. For some cases, e.g. Feature2DToolbox.DrawKeypoints, the UMat variant ("accelerated") will run orders of magnitude slower than the Mat variant.

In practice, I've only observed gains from use of UMat with GPU-accelerated feature detection algorithms (e.g. AKAZE) and nowhere else in this codebase.