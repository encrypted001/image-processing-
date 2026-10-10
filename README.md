# NMG Image Matching

This project compares two stone-slab images using local visual features and
geometric verification. It is designed for the NMG image-matching assignment,
where a query image must be compared with reference images without reading
labels from the filenames.

## Installation

Use Python 3.10 or newer and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project uses:

- OpenCV (`opencv-python`)
- NumPy
- pandas (available for dataset/result analysis)

## Project structure

```text
nmg/
├── match.py
├── test_match.py
├── requirements.txt
├── Intern_Assignment_Dataset/
│   ├── reference/                 # Five labelled reference images
│   ├── queries/                   # Twenty labelled query images
│   └── README.md                  # Dataset description and labels
└── data/images/
    ├── reference/                 # Dataset mirror
    └── queries/                   # Dataset mirror
```

The filenames in the supplied dataset contain labels for evaluation. The
matching pipeline uses image pixels, not those labels.

## Algorithm overview

`match.py` performs these steps:

1. Load both images with OpenCV.
2. Resolve the supplied paths and reject missing or undecodable files.
3. Resize each image so its longest side is at most 1000 pixels.
4. Keep the resized colour image for visualization and convert a copy to
   grayscale.
5. Detect SIFT keypoints and compute 128-value SIFT descriptors.
6. Match descriptors with `cv2.BFMatcher(cv2.NORM_L2)` and `knnMatch(k=2)`.
7. Apply Lowe's ratio test with an initial threshold of `0.75`.
8. Estimate a homography with `cv2.findHomography(..., cv2.RANSAC, 5.0)`.
9. Count RANSAC inliers, calculate a 0--100 confidence score, and report a
   classification band.
10. Draw only geometrically verified inlier matches.

CLAHE is implemented as an opt-in preprocessing experiment and is not enabled
by the command-line interface.

## Required commands

Run the CLI with two image paths and an output directory:

```powershell
python match.py --image-a <path-to-image-a> --image-b <path-to-image-b> --out output/
```

For the supplied dataset, an actual working example is:

```powershell
python match.py --image-a data/images/queries/img_001__B05B_lighting_lookalike.jpg --image-b data/images/reference/img_036__B01A_carrara_reference.jpg --out output/
```

The assignment's example command is also supported:

```powershell
python match.py --image-a data/images/img_001.jpg --image-b data/images/img_002.jpg --out output/
```

It will return a clear error if those exact files are not present. In the
current workspace, `data/images/img_002.jpg` is absent, so successful execution
of that exact example is not claimed.

Focused unit tests can be run with:

```powershell
python -m unittest test_match.py
```

To compare all 20 query images against all 5 reference images and write the
100 query-reference results to `results.csv`, run:

```powershell
python evaluate.py
```

The evaluator reads images only from `Intern_Assignment_Dataset/queries/` and
`Intern_Assignment_Dataset/reference/`. Query images are never compared with
other queries, and references are never compared with other references.

To open the interactive image viewer, run:

```powershell
python visualize.py
```

Choose a query and a reference, then select **Compare**. The viewer displays
the two images with RANSAC-verified feature matches and the confidence result.
It uses Python's built-in Tkinter interface, so no extra UI package is needed.

## Output

The command prints:

- Keypoints A and B
- Total descriptor matches
- Good matches after the ratio test
- RANSAC inliers
- Inlier ratio
- Confidence score out of 100
- Classification band (`SAME`, `SIMILAR`, or `DIFFERENT`)
- Processing time in milliseconds

The `--out` directory receives an image named like:

```text
<image-a-stem>_vs_<image-b-stem>_ransac_inliers.jpg
```

This visualization contains only matches accepted as RANSAC inliers. The
confidence score is a heuristic score, not a calibrated probability.

