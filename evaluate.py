"""Compare every query image with every reference image in the NMG dataset."""

import argparse
import csv
from pathlib import Path
import re
import sys

import match


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FIELDS = [
    "query",
    "reference",
    "query_piece",
    "reference_piece",
    "pair_type",
    "keypoints_a",
    "keypoints_b",
    "total_matches",
    "good_matches",
    "ransac_inliers",
    "inlier_ratio",
    "confidence_score",
    "classification",
]


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside a query or reference folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def piece_code(path: Path) -> str:
    """Read a dataset piece code for CSV reporting only (never for matching)."""
    found = re.search(r"__([A-Z]\d{2}[A-Z])_", path.name)
    return found.group(1) if found else ""


def pair_type(query: Path, reference: Path) -> str:
    """Describe the filename labels for later evaluation, not classification."""
    query_piece = piece_code(query)
    reference_piece = piece_code(reference)
    if not query_piece or not reference_piece:
        return "unknown"
    if query_piece == reference_piece:
        return "same"
    if query_piece[:3] == reference_piece[:3]:
        return "lookalike"
    return "different"


def compare_dataset(dataset_dir: Path, output_csv: Path) -> int:
    """Write one result row for each query-reference pair."""
    queries = image_files(dataset_dir / "queries")
    references = image_files(dataset_dir / "reference")
    if not queries:
        raise ValueError(f"No query images found in {dataset_dir / 'queries'}")
    if not references:
        raise ValueError(f"No reference images found in {dataset_dir / 'reference'}")

    # Compute each image's SIFT features once, then reuse them across pairs.
    features = {}
    for image_path in queries + references:
        grayscale, _, resolved_path = match.load_image(image_path)
        keypoints, descriptors = match.extract_sift_features(grayscale)
        features[resolved_path] = (keypoints, descriptors)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for query_path in queries:
            query_keypoints, query_descriptors = features[query_path.resolve()]
            for reference_path in references:
                reference_keypoints, reference_descriptors = features[
                    reference_path.resolve()
                ]
                candidate_matches, good_matches = match.match_sift_features(
                    query_descriptors,
                    reference_descriptors,
                )
                homography, inlier_mask, inliers, inlier_ratio = match.verify_geometry(
                    query_keypoints,
                    reference_keypoints,
                    good_matches,
                )
                score, classification = match.classify_geometry(
                    homography,
                    inlier_mask,
                    inliers,
                    inlier_ratio,
                )
                query_piece = piece_code(query_path)
                reference_piece = piece_code(reference_path)
                writer.writerow(
                    {
                        "query": str(query_path.relative_to(PROJECT_DIR)),
                        "reference": str(reference_path.relative_to(PROJECT_DIR)),
                        "query_piece": query_piece,
                        "reference_piece": reference_piece,
                        "pair_type": pair_type(query_path, reference_path),
                        "keypoints_a": len(query_keypoints),
                        "keypoints_b": len(reference_keypoints),
                        "total_matches": len(candidate_matches),
                        "good_matches": len(good_matches),
                        "ransac_inliers": inliers,
                        "inlier_ratio": f"{inlier_ratio:.6f}",
                        "confidence_score": f"{score:.2f}",
                        "classification": classification,
                    }
                )
                rows_written += 1
    return rows_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare every query only against the reference images."
    )
    parser.add_argument(
        "--dataset-dir",
        default="Intern_Assignment_Dataset",
        help="Dataset folder containing queries/ and reference/.",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="CSV file to write (default: results.csv).",
    )
    arguments = parser.parse_args()
    dataset_dir = Path(arguments.dataset_dir).expanduser()
    output_csv = Path(arguments.output).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_DIR / dataset_dir
    if not output_csv.is_absolute():
        output_csv = PROJECT_DIR / output_csv

    try:
        count = compare_dataset(dataset_dir.resolve(), output_csv.resolve())
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Compared {count} query-reference pairs.")
    print(f"Queries: {len(image_files(dataset_dir / 'queries'))}")
    print(f"References: {len(image_files(dataset_dir / 'reference'))}")
    print(f"Results: {output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
