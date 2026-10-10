"""Load and preprocess an image for the NMG image-matching project."""

from pathlib import Path
import argparse
import sys
import time

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = Path("Intern_Assignment_Dataset") / "queries" / "img_001.jpg"
DEFAULT_REFERENCE_IMAGE = (
    Path("Intern_Assignment_Dataset") / "reference" / "img_036.jpg"
)
DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_REPROJECTION_THRESHOLD = 5.0
DEFAULT_SAME_THRESHOLD = 70.0
DEFAULT_SIMILAR_THRESHOLD = 40.0
DEFAULT_DIFFERENT_THRESHOLD = 20.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(
    image_path: str | Path,
    max_size: int = 1000,
    use_clahe: bool = False,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load and preprocess an image.

    Relative paths are resolved from this script's directory. The returned
    colour image is resized but otherwise preserved for visualization, while
    the grayscale image is the preprocessing output for later matching.
    CLAHE is opt-in so its effect can be measured rather than assumed.
    """
    if max_size < 1:
        raise ValueError(f"max_size must be positive, got {max_size}")

    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()

    if not path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.parent.is_dir():
        # The supplied dataset appends labels to each original image ID, such as
        # img_001__B05B_lighting_lookalike.jpg. Match by ID without renaming it.
        matches = sorted(
            candidate
            for candidate in path.parent.rglob(f"{path.stem}__*{path.suffix}")
            if candidate.is_file()
        )
        if len(matches) == 1:
            path = matches[0].resolve()
        elif len(matches) > 1:
            names = "\n  ".join(str(candidate) for candidate in matches)
            raise FileNotFoundError(
                f"Image path {image_path!s} is not an exact filename, and its ID matches "
                f"multiple dataset files:\n  {names}\nPass one complete filename to choose an image."
            )

    if not path.is_file():
        raise FileNotFoundError(
            f"Image file does not exist: {path}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Expected the default dataset image at: {PROJECT_DIR / DEFAULT_IMAGE}\n"
            "Check that the dataset was extracted into Intern_Assignment_Dataset and that the filename matches."
        )

    colour_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if colour_image is None:
        raise ValueError(
            f"OpenCV found the file but could not decode it as an image: {path}. "
            "Check that the file is a supported, uncorrupted image."
        )

    height, width = colour_image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_size:
        scale = max_size / longest_side
        resized_dimensions = (round(width * scale), round(height * scale))
        colour_image = cv2.resize(
            colour_image, resized_dimensions, interpolation=cv2.INTER_AREA
        )

    grayscale_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayscale_image = clahe.apply(grayscale_image)

    return grayscale_image, colour_image, path


def extract_sift_features(
    grayscale_image: np.ndarray,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect SIFT keypoints and compute their descriptors."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grayscale_image, None)
    return keypoints, descriptors


def visualize_keypoints(
    colour_image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
) -> np.ndarray:
    """Draw detected keypoints on a colour image for debugging."""
    return cv2.drawKeypoints(
        colour_image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def match_sift_features(
    first_descriptors: np.ndarray | None,
    second_descriptors: np.ndarray | None,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> tuple[list[list[cv2.DMatch]], list[cv2.DMatch]]:
    """Match SIFT descriptors and filter candidates with Lowe's ratio test."""
    if not 0 < ratio_threshold <= 1:
        raise ValueError(
            f"ratio_threshold must be greater than 0 and at most 1, "
            f"got {ratio_threshold}"
        )
    if first_descriptors is None or second_descriptors is None:
        return [], []
    if len(first_descriptors) == 0 or len(second_descriptors) < 2:
        return [], []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidate_matches = matcher.knnMatch(
        first_descriptors,
        second_descriptors,
        k=2,
    )
    good_matches = [
        neighbours[0]
        for neighbours in candidate_matches
        if len(neighbours) >= 2
        and neighbours[0].distance < ratio_threshold * neighbours[1].distance
    ]
    return candidate_matches, good_matches


def visualize_matches(
    first_colour_image: np.ndarray,
    first_keypoints: list[cv2.KeyPoint],
    second_colour_image: np.ndarray,
    second_keypoints: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
) -> np.ndarray:
    """Draw ratio-test-passing matches for debugging."""
    return cv2.drawMatches(
        first_colour_image,
        first_keypoints,
        second_colour_image,
        second_keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def verify_geometry(
    first_keypoints: list[cv2.KeyPoint],
    second_keypoints: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    reprojection_threshold: float = DEFAULT_REPROJECTION_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray | None, int, float]:
    """Estimate a RANSAC homography and summarize its inlier support."""
    if reprojection_threshold <= 0:
        raise ValueError(
            "reprojection_threshold must be positive, "
            f"got {reprojection_threshold}"
        )
    if len(good_matches) < 4:
        return None, None, 0, 0.0

    source_points = np.float32(
        [first_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [second_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        reprojection_threshold,
    )
    if homography is None or inlier_mask is None:
        return None, None, 0, 0.0

    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / len(good_matches)
    return homography, inlier_mask, inlier_count, inlier_ratio


def homography_warnings(
    homography: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    first_keypoints: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    image_shape: tuple[int, ...],
) -> list[str]:
    """Report non-fatal signs that geometric verification may be unreliable."""
    if homography is None or inlier_mask is None:
        return ["Homography estimation failed or did not have enough correspondences."]
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-8:
        return ["Estimated homography is not a well-conditioned transformation."]

    inlier_points = np.float32(
        [
            first_keypoints[match.queryIdx].pt
            for match, is_inlier in zip(good_matches, inlier_mask)
            if is_inlier
        ]
    )
    if len(inlier_points) < 4:
        return ["Too few inliers remain for a reliable geometric decision."]

    height, width = image_shape[:2]
    region_area = cv2.contourArea(
        cv2.convexHull(inlier_points.reshape(-1, 1, 2))
    )
    if region_area < 0.01 * width * height:
        return ["Inliers are concentrated in less than 1% of the first image."]
    return []


def confidence_score(
    homography: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    inlier_count: int,
    inlier_ratio: float,
    count_scale: int = 100,
) -> float:
    """Return a 0-100 geometric confidence score, not a probability.

    The score gives equal weight to two signals:
    ``50 * min(inlier_count / count_scale, 1)`` rewards many inliers, and
    ``50 * inlier_ratio`` rewards a high proportion of consistent matches.
    ``count_scale`` is an explicit saturation point, not a learned constant.
    Any failed verification receives zero.
    """
    if count_scale < 1:
        raise ValueError(f"count_scale must be positive, got {count_scale}")
    if homography is None or inlier_mask is None:
        return 0.0
    if inlier_count <= 0 or not 0 <= inlier_ratio <= 1:
        return 0.0

    count_component = min(inlier_count / count_scale, 1.0)
    return round(50.0 * count_component + 50.0 * inlier_ratio, 2)


def classify_confidence(
    score: float,
    same_threshold: float = DEFAULT_SAME_THRESHOLD,
    similar_threshold: float = DEFAULT_SIMILAR_THRESHOLD,
    different_threshold: float = DEFAULT_DIFFERENT_THRESHOLD,
) -> str:
    """Classify a score using configurable lower bounds.

    These initial thresholds are engineering defaults only. They must be
    calibrated against labelled dataset results before being treated as a
    reliable classifier.
    """
    if not (
        0 <= different_threshold <= similar_threshold <= same_threshold <= 100
    ):
        raise ValueError(
            "thresholds must satisfy 0 <= different <= similar <= same <= 100"
        )
    if score >= same_threshold:
        return "SAME"
    if score >= similar_threshold:
        return "SIMILAR"
    if score <= different_threshold:
        return "DIFFERENT"
    # The interval between DIFFERENT and SIMILAR is deliberately conservative:
    # it remains SIMILAR until labelled evaluation supports a different policy.
    return "SIMILAR"


def classify_geometry(
    homography: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    inlier_count: int,
    inlier_ratio: float,
    same_threshold: float = DEFAULT_SAME_THRESHOLD,
    similar_threshold: float = DEFAULT_SIMILAR_THRESHOLD,
    different_threshold: float = DEFAULT_DIFFERENT_THRESHOLD,
) -> tuple[float, str]:
    """Score and classify verified geometry without claiming probability."""
    score = confidence_score(
        homography,
        inlier_mask,
        inlier_count,
        inlier_ratio,
    )
    label = classify_confidence(
        score,
        same_threshold,
        similar_threshold,
        different_threshold,
    )
    return score, label


def _debug_image_path(image_path: Path) -> Path:
    """Return a predictable path for a generated keypoint visualization."""
    return PROJECT_DIR / f"{image_path.stem}_sift_keypoints.jpg"


def _debug_matches_path(first_path: Path, second_path: Path) -> Path:
    """Return a predictable path for a generated match visualization."""
    return PROJECT_DIR / f"{first_path.stem}_vs_{second_path.stem}_ransac_inliers.jpg"


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Parse the two input images and output directory for the CLI."""
    parser = argparse.ArgumentParser(
        description="Compare two images with SIFT, ratio testing, and RANSAC."
    )
    parser.add_argument(
        "--image-a",
        default=str(DEFAULT_IMAGE),
        help=f"Path to the first image (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--image-b",
        default=str(DEFAULT_REFERENCE_IMAGE),
        help=f"Path to the second image (default: {DEFAULT_REFERENCE_IMAGE}).",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_DIR / "output"),
        help="Directory for the visualization (default: ./output).",
    )
    return parser.parse_args(arguments)


def main() -> int:
    try:
        arguments = _parse_arguments(sys.argv[1:])
    except SystemExit as error:
        return int(error.code)

    start_time = time.perf_counter()
    requested_path = Path(arguments.image_a)
    second_path = Path(arguments.image_b)
    try:
        first_image = load_image(requested_path)
        second_image = load_image(second_path)
        output_directory = Path(arguments.out).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: could not create output directory: {error}", file=sys.stderr)
        return 1

    first_keypoints, first_descriptors = extract_sift_features(first_image[0])
    second_keypoints, second_descriptors = extract_sift_features(second_image[0])
    candidate_matches, good_matches = match_sift_features(
        first_descriptors,
        second_descriptors,
    )
    homography, inlier_mask, inlier_count, inlier_ratio = verify_geometry(
        first_keypoints,
        second_keypoints,
        good_matches,
    )

    inlier_matches = (
        [
            match
            for match, is_inlier in zip(good_matches, inlier_mask)
            if is_inlier
        ]
        if inlier_mask is not None
        else []
    )
    match_visualization = visualize_matches(
        first_image[1],
        first_keypoints,
        second_image[1],
        second_keypoints,
        inlier_matches,
    )
    match_visualization_path = output_directory / _debug_matches_path(
        first_image[2],
        second_image[2],
    ).name
    if not cv2.imwrite(str(match_visualization_path), match_visualization):
        print(
            f"Error: could not write match visualization: {match_visualization_path}",
            file=sys.stderr,
        )
        return 1

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Keypoints A: {len(first_keypoints)}")
    print(f"Keypoints B: {len(second_keypoints)}")
    print(f"Total matches: {len(candidate_matches)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"RANSAC inliers: {inlier_count}")
    print(f"Inlier ratio: {inlier_ratio:.3f}")
    score, classification = classify_geometry(
        homography,
        inlier_mask,
        inlier_count,
        inlier_ratio,
    )
    print(f"Confidence score: {score:.2f}/100")
    print(f"Classification band: {classification}")
    for warning in homography_warnings(
        homography,
        inlier_mask,
        first_keypoints,
        good_matches,
        first_image[0].shape,
    ):
        print(f"Geometry warning: {warning}")
    print(f"Match visualization: {match_visualization_path}")
    print(f"Processing time: {processing_time_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
