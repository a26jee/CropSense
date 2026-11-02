# CropSense — Complete ML Crop Recommendation Website

A responsive single-page crop recommendation website powered by a trained scikit-learn classifier and a Flask API.

## Included
- Responsive HTML/CSS/JavaScript frontend
- Flask backend serves the website and JSON API
- Six trained classifiers plus the selected `best_model.pkl`
- Prediction with up to three model-estimated class probabilities
- Dashboard: model comparison, test metrics, dataset class distribution, confusion matrix and training workflow
- The model loads once at server startup; no model retraining occurs for each prediction

## Project layout
```text
CropSense/
├── backend/app.py
├── backend/models/       # trained .pkl files and model_metadata.json
├── dataset/Crop_recommendation.csv
├── frontend/templates/index.html
├── frontend/static/app.js
├── frontend/static/style.css
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Windows setup
Open a terminal in the extracted `CropSense` folder:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python backend/app.py
```
Open **http://127.0.0.1:5000** in your browser.

## macOS / Linux setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python backend/app.py
```
Then visit **http://127.0.0.1:5000**.

## Retrain
The included models are ready to use. To retrain after changing the CSV or training script, run `python train.py` from the project root, then restart Flask. The best model is selected by mean macro F1 from 5-fold stratified cross-validation on the training partition; the held-out test split is reserved for final reporting.

## API
- `GET /api/stats` — dataset and model evaluation information.
- `POST /api/predict` — expects JSON with `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`.

Example:
```json
{"N":90,"P":42,"K":43,"temperature":24.5,"humidity":80,"ph":6.5,"rainfall":200}
```

## Notes
Enter measurements using the units shown: N/P/K in the dataset's nutrient units, temperature in °C, humidity in %, pH unitless, and rainfall in mm. The displayed class probabilities are model estimates, not necessarily calibrated probabilities. This educational benchmark is not a guarantee of crop suitability or yield; consider local season, irrigation, soil testing and agricultural expertise. Only load pickle files from trusted sources and use compatible Python/scikit-learn versions.
