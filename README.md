# Vehicle Theft Monitor

A local video-review prototype for parking-area surveillance. It detects common person/vehicle classes, tracks them inside a single clip, and shows heuristic interaction risk events. It does not identify people by name, prove car ownership, or send notifications.

## Run

1. Install Python 3.10+ and create a virtual environment.
2. Run pip install -r requirements.txt
3. Run streamlit run app.py
4. Upload an MP4, MOV, AVI, or MKV and choose Analyze. First use of the default Ultralytics model may download weights.

## Current prototype

- YOLO object detection and ByteTrack track IDs within one video.
- Video-level NORMAL/DETECTED label from peak heuristic risk.
- Risk events: person near vehicle, person/vehicle box overlap (possible entry or occlusion), and vehicle motion when no person is nearby.
- Track IDs can switch and are not persistent between videos or cameras. Association to an owner needs manual confirmation and context.
- Camera movement, occlusion, and imperfect detections can cause false alerts or missed events. Treat outputs as prompts for human review.

## Training data and split

Organize labeled recordings as data/videos/normal and data/videos/detected. Split by original recording/session to avoid frames from one recording leaking into multiple sets. The selected split is 70% train, 15% validation, 15% test. This initial version does not yet train a theft classifier: collect representative labeled clips and define the target events before training/evaluation can be meaningful. The upload app runs its heuristic baseline without a trained model.

## To extend into a real system

Annotate owner-confirmed parking, approaches, vehicle entry, departure, and vehicle movement with timestamps; train/evaluate against complete held-out recordings; measure missed events and false alarms; add explicit owner enrollment and a secure notification service only after selecting a delivery channel. Do not treat visual appearance matching as proof of identity or ownership.


## Train the clip classifier

Put videos into data/videos/normal and data/videos/detected (at least 10 videos per class). Then run python train.py. The script extracts detector-event features, makes a stratified approximate 70/15/15 train/validation/test split, selects a small k-nearest-neighbor baseline in pure Python, reports metrics on the held-out test split, and writes models/theft_knn.json. The upload app uses this model automatically when it exists; otherwise it reports its heuristic result. Validation is held out from fitting in this baseline, but hyperparameter selection is not implemented. Keep recordings from the same camera session together to prevent leakage; this starter split is per-file and must be adapted/grouped for related clips. Training runs the object detector over every video and can take substantial time; use a GPU or sample a manageable dataset.


If an older requirements file installed scikit-learn, it is no longer needed by this project. Install the updated requirements with pip install -r requirements.txt. The training classifier uses only Python standard-library code, avoiding the blocked scikit-learn native extension.

See [DATASET_SOURCES.md](DATASET_SOURCES.md) for 20 Pexels review candidates, the EPFL research dataset instructions, and rules for creating trustworthy normal/detected labels. The videos themselves have not been downloaded into this workspace; Pexels clips must be downloaded and reviewed, and EPFL material must be obtained under its research-use terms.

After a successful training run, analyzing a video in Streamlit shows the saved held-out test metrics below the event timeline, including accuracy, precision, recall, F1, specificity, confusion matrix, and per-class scores. These are model evaluation scores from labeled test clips, not measurements for the current unlabeled upload. Re-run training after adding/changing data to refresh them.

## Annotated video output

After choosing Analyze video, the app renders a downloadable MP4. Person boxes are labeled PERSON #ID; car/truck/bus/motorcycle boxes are labeled VEHICLE #ID. Nearby person/vehicle tracks are connected with a line and their IDs. The status overlay shows frame number, visible track counts, current/peak heuristic risk, and event count. The tracker IDs are local to that clip and can switch after occlusion. The sidebar sampling control runs detection every Nth frame while writing all source frames, reusing the latest boxes between detections.

## Registering an owner person/car pair

Tracker IDs for people and vehicles are separate, so the raw numbers should not be compared for equality. First analyze the clip with both owner ID inputs set to 0. Read the Person and Vehicle track IDs from the result, then enter the known owner's person ID and their vehicle ID in the sidebar and analyze again. The manually registered pair is drawn green with a connecting line; a different person track near that registered vehicle is drawn red and marked DETECTED. This depends on a human-confirmed pairing and is only valid within that clip; the tracker does not recognize an owner across separate uploads.
