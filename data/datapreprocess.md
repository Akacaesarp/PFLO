# PFLO Data Preprocessing Tool

This tool is designed for the PFLO (Pose Estimation Model of Field Maize Based on YOLO Architecture) project, used for processing annotation data and generating suitable inputs for training YOLO architecture models.

## Overview

The tool provides three main functions:

1. **Image Extraction**: Extract image data from Labelme format JSON files and save as JPG files
2. **Annotation Conversion**: Convert Labelme JSON annotations to YOLO format TXT annotation files
3. **Keypoint Bias Reduction Processing**: Sort and interpolate manually annotated keypoints in JSON files to generate uniformly distributed keypoints

## Dependencies

Ensure the following Python dependencies are installed:

```bash
pip install pillow tqdm numpy
```

## Usage

### Command Line Arguments

The tool supports the following command line arguments:

```
usage: python preprocess.py [-h] --task {extract,convert,process,all} --json-dir JSON_DIR --save-dir SAVE_DIR
                           [--extract-dir EXTRACT_DIR] [--convert-dir CONVERT_DIR] [--process-dir PROCESS_DIR]
                           [--ratio RATIO] [--num-points NUM_POINTS] [--num-pad NUM_PAD]

Data Preprocessing Tool

optional arguments:
  -h, --help            Show help information and exit
  --task {extract,convert,process,all}
                        Task to execute: extract (extract images), convert (convert annotations), 
                        process (process points), all (all tasks)
  --json-dir JSON_DIR   JSON files directory
  --save-dir SAVE_DIR   Save directory
  --extract-dir EXTRACT_DIR
                        Directory to save extracted images (only needed when task=all)
  --convert-dir CONVERT_DIR
                        Directory to save converted annotations (only needed when task=all)
  --process-dir PROCESS_DIR
                        Directory to save processed points (only needed when task=all)
  --ratio RATIO         Scaling ratio (default: 1)
  --num-points NUM_POINTS
                        Number of points to select (default: 10)
  --num-pad NUM_PAD     Number of additional points between each pair of points during interpolation (default: 10)
```

### Example Usage

#### 1. Extract Images Only

```bash
python preprocess.py --task extract --json-dir /path/to/json/files --save-dir /path/to/save/images --ratio 4
```

This command extracts images from JSON files and saves them to the specified directory, while resizing the images to 1/4 of their original size.

#### 2. Convert Annotations Only

```bash
python preprocess.py --task convert --json-dir /path/to/json/files --save-dir /path/to/save/labels --ratio 4
```

This command converts Labelme JSON annotations to YOLO format TXT annotations, adjusting coordinates to match the resized images.

#### 3. Process Keypoints Only

```bash
python preprocess.py --task process --json-dir /path/to/json/files --save-dir /path/to/save/processed/json --num-points 17 --num-pad 10
```

This command processes keypoints in JSON files by sorting and interpolating, generating 17 uniformly distributed keypoints for each target.

#### 4. Execute All Tasks

```bash
python preprocess.py --task all --json-dir /path/to/json/files --extract-dir /path/to/save/images --convert-dir /path/to/save/labels --process-dir /path/to/save/processed/json --ratio 4 --num-points 17 --num-pad 10
```

This command executes all three tasks in sequence.

## Detailed Functions

### Image Extraction (`json2jpg`)

This function extracts image data (base64 encoded) from Labelme format JSON files and saves them as JPG files. Processing includes:

- Decoding base64 image data
- Automatically rotating images to the correct orientation based on EXIF information
- Resizing images according to the specified ratio
- Preserving original file modification times

### Annotation Conversion (`convert_label_json`)

This function converts Labelme JSON annotations to YOLO format annotation files. Main processing:

- Converting bounding box coordinates from (xmin, ymin, xmax, ymax) to YOLO-required (x_center, y_center, width, height) format
- Adjusting coordinates according to the specified scaling ratio
- Distinguishing between different categories of leaves and stalks (category index: stalk is 0, leaf is 1)
- Including keypoint coordinate information

### Keypoint Processing (`sort_and_interpolate_points`)

This function processes keypoints in JSON files by sorting and interpolating, making keypoints uniformly distributed across objects:

- Sorting stalk keypoints by y-coordinates from top to bottom
- Sorting leaf keypoints by distance from the root
- Performing linear interpolation between adjacent keypoints to increase point count
- Selecting a specified number of uniformly distributed points

## Data Preprocessing Method and Results

Our data preprocessing method addresses inconsistencies in manual annotations through a three-step process:

1. **Step 1: Sort** - Sort the originally annotated keypoints by ground truth positions
2. **Step 2: Insert** - Insert additional points through linear interpolation between existing keypoints
3. **Step 3: Sample** - Create uniformly sampled points along the plant structure

Additionally, our bounding box generation utilizes minimum enclosing rectangles with optimized padding (25 pixels) to provide adequate contextual information.

![Data Preprocessing Workflow](https://github.com/Akacaesarp/PFLO/blob/master/figure/3.png)

The comparison before and after preprocessing shows significant improvement in keypoint distribution and consistency:

![Before and After Preprocessing](https://github.com/Akacaesarp/PFLO/blob/master/figure/4.svg)

### Quantitative Improvement

**Table 1.** Comparison of Manual and Semi-automatic Annotation Metrics for Each Leaf or Stalk

| Data Stage | Number of Keypoints (Mean ± Std) | Inter-point Distance (pixels, Mean ± Std) |
|------------|----------------------------------|-------------------------------------------|
| Manual Annotation (Before) | 6.36 ± 2.90 | 176.65 ± 129.11 |
| Semi-automatic (After) | 17.00 ± 0.00 | 49.92 ± 30.97 |

The preprocessing significantly reduces annotation variability, providing consistent keypoint count (standard deviation reduced from 2.90 to 0) and more uniform inter-point distances (reduced from 176.65 to 49.92 pixels on average).

## Notes

1. This tool assumes that input JSON files are generated using the Labelme annotation tool
2. To avoid duplicate processing, the tool will skip files if the target file already exists
3. For cases with fewer than two keypoints, the tool will provide default processing but will issue warnings
4. When processing large amounts of data, it is recommended to test a small sample first to ensure the results meet expectations

## Implementation Details

- Sorting strategy: For stalks, sort by vertical position (y-coordinate); for leaves, sort by distance from the stalk
- Interpolation method: Use linear interpolation between adjacent points to increase the number of intermediate points
- Standardization: Keep the final number of keypoints consistent for easier model training

## Related Links

- PFLO project: https://github.com/Akacaesarp/PFLO
- Dataset: http://phenomics.agis.org.cn/#/category (MIPDB database)
