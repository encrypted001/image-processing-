# NMG Image-Matching Report

## Scope and evaluation method

The supplied labelled dataset contains five reference images and twenty query
images, giving 100 reference/query comparisons:

- 10 same-piece pairs
- 10 sibling lookalike pairs
- 80 pairs from different blocks

For this report, the relation was determined from the dataset's piece-code
labels only for evaluation. The matching pipeline itself received image data
and did not read those labels.

All results below were measured by running the current functions in
`match.py` on the 100 dataset pairs with the default parameters. No accuracy
claim is made beyond this dataset and this exact configuration.

## 1. Algorithm selection and parameters

### Image preprocessing

- OpenCV image loading in colour mode.
- Longest image side resized to at most 1000 pixels.
- Grayscale conversion for feature extraction.
- Original resized colour image retained for visualizations.
- CLAHE is implemented but disabled by default; its effect was not included in
  the reported evaluation.

### Local features and descriptor matching

- SIFT via `cv2.SIFT_create()`.
- Brute-force descriptor matching via
  `cv2.BFMatcher(cv2.NORM_L2)`.
- Two nearest neighbours requested with `knnMatch(k=2)`.
- Lowe ratio threshold: `0.75`.

SIFT and the L2 norm are appropriate together because SIFT descriptors are
floating-point vectors. The ratio test rejects ambiguous nearest-neighbour
matches before geometric verification.

### Geometric verification

- Homography estimated with `cv2.findHomography`.
- RANSAC enabled.
- Reprojection threshold: `5.0` pixels.
- At least four good correspondences are required.
- A failed estimate returns zero inliers and zero confidence.

## 2. Confidence formula and threshold selection

The implementation uses this documented heuristic:

```text
count_component = min(inlier_count / 100, 1.0)

confidence =
    50 * count_component
  + 50 * inlier_ratio
```

The score is bounded conceptually to 0--100 and is not a calibrated
probability. The count scale of 100 is a saturation point chosen as an initial
engineering parameter.

The current classification thresholds are:

| Band | Rule |
|---|---|
| `SAME` | score >= 70 |
| `SIMILAR` | 40 <= score < 70 |
| `DIFFERENT` | score <= 20 |
| Conservative gap | 20 < score < 40 is reported as `SIMILAR` |

These thresholds were not selected from a formal training/validation split.
They are initial defaults and must be calibrated using labelled dataset score
distributions. In particular, a calibration procedure should preserve zero
lookalike pairs classified as `SAME` while recovering as many true same-piece
pairs as possible.

## 3. Measured results by pair type

The following values are the measured minimum, maximum, and mean for each
relation category.

| Pair type | Count | Inliers min/max/mean | Inlier ratio min/max/mean | Score min/max/mean |
|---|---:|---:|---:|---:|
| Same piece | 10 | 5 / 5112 / 945.2 | 0.600 / 1.000 / 0.889 | 33.00 / 99.94 / 65.28 |
| Lookalike sibling | 10 | 0 / 5 / 3.2 | 0.000 / 1.000 / 0.335 | 0.00 / 52.00 / 18.36 |
| Different block | 80 | 0 / 19 / 2.34 | 0.000 / 1.000 / 0.181 | 0.00 / 52.00 / 10.21 |

The maximum ratio of `1.000` in a non-same group does not by itself imply a
correct match: a small number of matches can all be RANSAC inliers. This is
one reason the score uses both inlier count and ratio.

## 4. Same-object classification performance

| Result | Count |
|---|---:|
| Same-piece pairs classified `SAME` | 3 / 10 |
| Same-piece pairs classified not `SAME` | 7 / 10 |

The measured same-object `SAME` rate on this dataset/configuration is therefore
3/10 (30%). This is an observed result, not a general accuracy claim. The
current threshold is conservative relative to several true same-piece scores.

## 5. Different-object rejection performance

For all non-same pairs (lookalikes plus different blocks):

| Result | Count |
|---|---:|
| Non-same pairs classified `SAME` | 0 / 90 |
| Non-same pairs not classified `SAME` | 90 / 90 |

The measured false-`SAME` count is zero for this evaluation. This does not
establish performance on images outside the supplied dataset.

## 6. Lookalike pairs wrongly classified `SAME`

```text
0 / 10 lookalike pairs were classified SAME.
```

Three lookalike pairs were classified `SIMILAR`, and seven were classified
`DIFFERENT`. A `SIMILAR` result is intentionally not treated as proof that the
physical object is the same.

## 7. Actual measured failure examples

The following are true same-piece pairs that failed to reach the current
`SAME` band:

### Example 1: B03B viewpoint change

```text
Query:     img_012__B03B_viewpoint_same.jpg
Reference: img_044__B03B_botticino_reference.jpg
Good matches: 10
RANSAC inliers: 6
Inlier ratio: 0.600
Score: 33.00
Classification: SIMILAR
```

The viewpoint change produced few ratio-test-passing matches. Six geometric
inliers were not enough to reach the `SAME` threshold.

### Example 2: B01A rotation and scale change

```text
Query:     img_021__B01A_rotation_scale_same.jpg
Reference: img_036__B01A_carrara_reference.jpg
Good matches: 7
RANSAC inliers: 5
Inlier ratio: 0.714
Score: 38.21
Classification: SIMILAR
```

This pair retained only seven candidate matches after the ratio test. Although
five were geometrically consistent, the low inlier count kept the heuristic
score below the `SIMILAR` boundary's upper threshold and therefore below
`SAME`.

### Additional measured non-same ambiguity

This pair was not a false `SAME`, but it illustrates why the intermediate band
needs calibration:

```text
Query:     img_046__B02B_rotation_scale_lookalike.jpg
Reference: img_039__B02A_statuario_reference.jpg
Good matches: 6
RANSAC inliers: 5
Inlier ratio: 0.833
Score: 44.17
Classification: SIMILAR
```

The sibling lookalike has enough local/geometric agreement to enter the
intermediate band, but not enough evidence under the current rules to be
called `SAME`.

## 8. Limitations and possible improvements

### Measured limitations

- Seven of ten true same-piece pairs were not classified `SAME` with the
  current score thresholds.
- Twenty-four non-same pairs were classified `SIMILAR` rather than
  `DIFFERENT` (three lookalikes and twenty-one different-block pairs).
- Inliers can be concentrated in a small region or arise from repeated
  texture; the implementation reports warnings, but the current score does
  not reject every such case.
- A homography is a useful model for approximately planar appearance changes,
  but it is not proof of physical identity.

### Possible improvements

- Calibrate `SAME`, `SIMILAR`, and `DIFFERENT` thresholds from labelled score
  distributions using a documented validation procedure.
- Tune the ratio and reprojection thresholds jointly instead of evaluating
  only the initial `0.75` and `5.0` values.
- Require a stronger minimum inlier count before allowing `SAME`.
- Add spatial coverage checks so matches must cover a meaningful portion of
  the image, not just one small texture patch.
- Compare alternative feature extractors and preprocessing variants, including
  the currently untested CLAHE option.
- Evaluate multiple references per stone type and more varied viewpoints,
  lighting, occlusion, and scale changes.
- Add a reproducible full-evaluation script that writes per-pair CSV results
  and visual diagnostics.

The CLAHE experiment, threshold sweep, alternative algorithms, and
out-of-dataset generalization were not run for this report. Claims about those
experiments remain untested.

