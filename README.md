# Indian Cricket Player Face Identification

A local Streamlit application that detects one face with OpenCV YuNet, extracts
pretrained OpenCV SFace embeddings, and identifies a supported player with a
scikit-learn linear SVM. The pretrained face networks are not trained from
scratch. A SFace cosine-reference matcher remains as a fallback if no SVM artifact
exists.

## Dataset inspection

The extracted dataset currently lives at `indian cricketer/` (with a space; no
`indian_cricketer/` directory exists). The immediate folder names are the labels;
the inspection code preserves them as written, converting underscores to spaces
only when present. The root contains 324 player folders and no nested class
folders, empty folders, or root-level files. It contains 7,687 files total:

- 7,079 `.jpg`, 351 `.jpeg`, 236 `.png`
- 17 `.gif`, 3 `.jfif`, and 1 `.bmp`
- 7,657 files with supported still-image extensions decoded successfully
- 14 files were unreadable: 13 JPGs and one GIF
- 17 GIF files are unsupported; the remaining 3 JFIF and 1 BMP files are decoded
  and included by the current pipeline
- File-count distribution: 242 classes have 25 files, 79 have 20, 2 have 24, and
  1 has 9. Ravi Shastri's 9 JPG files were all unreadable in the scan.

The full per-player total/readable/unsupported counts are recorded in
`dataset_inventory.csv`. Rebuild that inventory with:

```powershell
python inspect_dataset.py --dataset-dir ".\indian cricketer" --csv dataset_inventory.csv
```

The dataset folder contains no license, readme, citation, source URL, or terms
file. The extracted dataset is credited to **Avirat Gupta** and its Kaggle page is
[Indian Cricketers Dataset](https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset).
The Kaggle page displays **CC0: Public Domain**. That page-level declaration is
not independent verification of copyright or usage rights for every third-party
photograph contained in the dataset; those individual rights have not been
independently verified. Do not claim that every photo is cleared for reuse or
redistribution. Review the Kaggle page and any source-photo terms before
publishing dataset images or a model trained on them. The project does not
download or scrape more photos. The original `reference_images/` directories
remain supported and are not overwritten.

`dataset_inventory.csv` is generated from this unverified dataset and is ignored
by Git. Do not publish it until the original dataset terms have been checked. To
reproduce results elsewhere, obtain the same dataset directly from its verified
original source only if its license permits your use. Otherwise, substitute your
own or separately licensed, consented folder-labeled photos. Do not copy the local
dataset from this project or assume Kaggle hosting grants redistribution rights.

## Dataset layout and training

Training accepts a folder-per-class dataset. Existing reference photos continue
to work with the default `reference_images/` input; pass the extracted dataset
explicitly to train on its folder labels:

```text
indian cricketer/
  Jasprit Bumrah/
    Image_1.jpg
  Rohit Sharma/
    Image_1.jpg
```

The extractor scans each class folder recursively, decodes supported `.jpg`,
`.jpeg`, `.png`, `.webp`, `.jfif`, and `.bmp` images, skips unreadable files and
images without exactly one detected face, and uses YuNet/SFace exactly as the app
does. GIF is excluded because it may be animated and is not reliably decoded by
the current OpenCV image path.

Exact and perceptual near-duplicates are assigned to one group before splitting.
The default pHash threshold is 12 bits out of 63 and comparisons are limited to
images with the same folder label, avoiding accidental cross-player grouping due
to similar backgrounds. Every copy of a byte-identical image assigned conflicting
labels is excluded and reported; the original files are not modified. pHash
remains a heuristic and may miss crops, edits, or frames from the same session.
Across labels, a pair is excluded only when whole-image pHash distance is at most
12 and SFace cosine similarity is at least 0.965; this catches near-identical
face crops while reducing background-only false matches. Review image provenance
manually. Per-class deterministic
allocation keeps groups intact while ensuring each
eligible class appears in train, validation, and test. Classes with fewer than
four distinct groups are explicitly excluded; at least two eligible classes are
needed to continue. If any required split is invalid, no test metrics are
reported and no replacement classifier is saved.

From PowerShell in the project folder, after installing the requirements and
downloading the YuNet/SFace model files described below:

```powershell
python inspect_dataset.py --dataset-dir ".\indian cricketer"
python validate_setup.py
python train_classifier.py --dataset-dir ".\indian cricketer" --seed 42 --workers 2
python train_classifier.py --dataset-dir ".\indian cricketer" --evaluate-only --workers 2
```

The training command prints actual held-out accuracy, macro precision, recall,
F1, split sizes, excluded classes, and a labeled confusion matrix. Artifacts are
written to `models/artifacts/player_svm.joblib`, `metadata.json`, and
`confusion_matrix.csv`. The matrix includes an `Unknown player` prediction column.
The app reads the supported class names from the trained model artifact. If no
classifier is available, it shows reference-folder labels and the cosine matcher.
Because Streamlit caches the classifier, stop the running app with Ctrl+C and
restart it after training to load the updated roster/artifact.

Reproduce the current run from the project directory:

```powershell
python inspect_dataset.py --dataset-dir ".\indian cricketer" --csv dataset_inventory.csv
python train_classifier.py --dataset-dir ".\indian cricketer" --seed 42 --workers 2
python train_classifier.py --dataset-dir ".\indian cricketer" --evaluate-only --workers 2
streamlit run app.py
```

The trained roster contains only classes with at least four distinct usable photo
groups after face detection, corrupt/unsupported file filtering, and duplicate
grouping. Check the CLI's `[EXCLUDED]` messages and artifact metadata to learn the
actual classes and images used; the raw folder count alone does not establish
support.

## Training results and limitations

Current actual run (seed 42, this dataset and the installed OpenCV models):

- 6,398 images yielded one detected face and an SFace embedding.
- 23 files in 11 byte-identical cross-label groups were excluded; 1,259 had no
  single detectable face, 13 supported still images were unreadable, and 17 GIFs
  were unsupported.
- 4 more files in 2 visually checked near-duplicate pairs across labels were
  excluded by the combined pHash/SFace rule.
- 6,371 images from 323 player labels were used (7-25 per label; median 20),
  with no remaining label excluded for insufficient groups.
- Grouped split: 3,829 train, 1,273 validation, and 1,269 test images across
  6,270 duplicate groups.
- Held-out accuracy **72.73% (0.7273)**, macro precision **0.8421**, macro recall
  **0.6933**, macro F1 **0.7414**. The confusion matrix is 324 x 324, including
  `Unknown player`.
- Validation-derived rejection margin: **0.425620**.

`--evaluate-only` reproduced these metrics. It refuses to evaluate if labels,
image paths, duplicate groups, or embeddings differ from the training dataset.
The artifact metadata stores the per-class counts and dataset fingerprint. These
scores represent one split of this dataset, not guaranteed future-photo accuracy.

The integrated YuNet/SFace/SVM path was also exercised on three held-out images
from Aashish Kapoor, Aavishkar Salvi, and Abey Kuruvilla; all three matched their
folder labels. This smoke check is not a separate performance estimate.

The dataset has 324 folder-derived labels and at most 25 files per class; several
folders contain 20. Some files are unreadable, some images do not contain exactly
one face, and pHash grouping can reduce usable independent samples. Per-class
test sets are small, so metrics can be unstable and do not establish performance
on new photos. Held-out known-player results do not measure unknown-person false
acceptance; the SVM margin threshold is a validation-derived heuristic, not a
calibrated probability or proof of identity.

Although the raw dataset has 324 player folders, the trained roster has 323
classes. All nine Ravi Shastri JPGs were unreadable, so no valid SFace samples
exist for that folder. Supporting that 324th label requires correcting or
replacing those source images with properly licensed, correctly labeled face
photos and retraining; the project does not fabricate or infer missing examples.

## Streamlit application

Launch from the project folder:

```powershell
streamlit run app.py
```

The supported-player list is loaded from the trained SVM labels when available.
The app requires exactly one detected face and displays the predicted player or
`Unknown player`. Similarity and SVM decision margins are scores, not probabilities.
Upload formats include the supported still-image extensions listed above.

## Models and setup

The required official OpenCV Zoo weights are:

- `models/face_detection_yunet_2023mar.onnx`
- `models/face_recognition_sface_2021dec.onnx`

Download from the official OpenCV Zoo model pages ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface))
using their verified official model asset URLs:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
  -OutFile ".\models\face_detection_yunet_2023mar.onnx"
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" `
  -OutFile ".\models\face_recognition_sface_2021dec.onnx"
```

The YuNet model folder states MIT licensing; the SFace model folder states
Apache-2.0 licensing. Review those license files for your intended use. Model
weights, SVM artifacts, and cached embeddings are excluded from Git. The dataset
directories `indian cricketer/` and `indian_cricketer/`, and user-supplied
`reference_images/`, are also ignored.

Install the Python dependencies in a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tests

Run all offline and synthetic tests, then compile all Python files:

```powershell
python -m pytest -q
python -m py_compile app.py face_id.py dataset.py inspect_dataset.py training.py train_classifier.py validate_setup.py tests\test_face_id.py tests\test_dataset.py tests\test_training.py
```

Automated synthetic tests do not claim recognition performance. The actual
training/evaluation command is the source of truth for any reported dataset
metrics. Keep source/license notes for photos and do not publish data unless its
license permits it.