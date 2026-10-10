"""Small Tkinter interface for visually inspecting query/reference matches."""

import base64
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

import match


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Intern_Assignment_Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PREVIEW_WIDTH = 1180
PREVIEW_HEIGHT = 650


def list_images(folder: Path) -> list[Path]:
    """List image files in a dataset folder in a stable order."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class MatchViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NMG Image Match Viewer")
        self.root.geometry("1240x820")
        self.feature_cache = {}

        self.queries = list_images(DATASET_DIR / "queries")
        self.references = list_images(DATASET_DIR / "reference")
        if not self.queries or not self.references:
            raise FileNotFoundError(
                f"Expected query and reference images in {DATASET_DIR}. "
                f"Found {len(self.queries)} queries and {len(self.references)} references."
            )

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Query image:").grid(row=0, column=0, sticky="w")
        self.query_choice = ttk.Combobox(
            controls,
            state="readonly",
            width=57,
            values=[path.name for path in self.queries],
        )
        self.query_choice.grid(row=0, column=1, padx=(6, 16), sticky="ew")
        self.query_choice.current(0)

        ttk.Label(controls, text="Reference image:").grid(row=0, column=2, sticky="w")
        self.reference_choice = ttk.Combobox(
            controls,
            state="readonly",
            width=52,
            values=[path.name for path in self.references],
        )
        self.reference_choice.grid(row=0, column=3, padx=6, sticky="ew")
        self.reference_choice.current(0)

        self.compare_button = ttk.Button(
            controls,
            text="Compare",
            command=self.compare,
        )
        self.compare_button.grid(row=0, column=4, padx=(12, 0))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        self.summary = tk.StringVar(value="Choose a query and reference image.")
        ttk.Label(root, textvariable=self.summary, padding=(10, 4)).pack(fill="x")

        self.preview = ttk.Label(root, anchor="center")
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        self.photo = None

        self.compare()

    def _features(self, path: Path):
        if path not in self.feature_cache:
            grayscale, colour, resolved_path = match.load_image(path)
            keypoints, descriptors = match.extract_sift_features(grayscale)
            self.feature_cache[path] = (
                grayscale,
                colour,
                resolved_path,
                keypoints,
                descriptors,
            )
        return self.feature_cache[path]

    def compare(self) -> None:
        query_path = self.queries[self.query_choice.current()]
        reference_path = self.references[self.reference_choice.current()]
        self.compare_button.configure(state="disabled")
        self.summary.set("Comparing images...")
        self.root.update_idletasks()

        try:
            query = self._features(query_path)
            reference = self._features(reference_path)
            candidate_matches, good_matches = match.match_sift_features(
                query[4], reference[4]
            )
            homography, inlier_mask, inlier_count, inlier_ratio = match.verify_geometry(
                query[3], reference[3], good_matches
            )
            score, classification = match.classify_geometry(
                homography, inlier_mask, inlier_count, inlier_ratio
            )

            inlier_matches = (
                [
                    item
                    for item, is_inlier in zip(good_matches, inlier_mask)
                    if is_inlier
                ]
                if inlier_mask is not None
                else []
            )
            combined = match.visualize_matches(
                query[1], query[3], reference[1], reference[3], inlier_matches
            )
            height, width = combined.shape[:2]
            scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height, 1.0)
            if scale < 1.0:
                combined = cv2.resize(
                    combined,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            encoded_ok, encoded_image = cv2.imencode(".png", combined)
            if not encoded_ok:
                raise ValueError("OpenCV could not prepare the match preview.")

            image_data = base64.b64encode(encoded_image.tobytes()).decode("ascii")
            self.photo = tk.PhotoImage(master=self.root, data=image_data)
            self.preview.configure(image=self.photo)
            self.summary.set(
                f"{classification}  |  Confidence {score:.2f}/100  |  "
                f"Keypoints {len(query[3]):,} / {len(reference[3]):,}  |  "
                f"Good matches {len(good_matches):,}  |  "
                f"RANSAC inliers {inlier_count:,} ({inlier_ratio:.1%})"
            )
        except (FileNotFoundError, ValueError, cv2.error) as error:
            self.preview.configure(image="")
            self.photo = None
            self.summary.set("Could not compare these images.")
            messagebox.showerror("Image comparison failed", str(error), parent=self.root)
        finally:
            self.compare_button.configure(state="normal")


def main() -> None:
    root = tk.Tk()
    try:
        MatchViewer(root)
    except (FileNotFoundError, OSError) as error:
        messagebox.showerror("NMG dataset unavailable", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
