# AI-Powered Student Career Prediction & Analytics

This upgraded project turns a small placement predictor into a full final-year mega project with: EDA, multi-model training, salary & role prediction, a Streamlit dashboard, dataset generator, and artifact persistence.

## Features
- Upload dataset and explore interactive EDA (Plotly)
- Train multi-model placement classifier and salary regressor (train_models.py)
- Persist models and reuse in Streamlit app
- Predict placement probability, salary estimate, and job role
- Rich synthetic dataset generator (generate_dataset.py)

## Run
1. `pip install -r requirements.txt`
2. (optional) `python generate_dataset.py` to create `data/student_placement_dataset.csv`
3. `python train_models.py` to train and save artifacts to `artifacts/`
4. `streamlit run app.py`

## Project Ideas / Future Scope
- Add SHAP-based explainability for individual predictions
- Deploy on Heroku / GCP / Render with a simple Dockerfile
- Add authentication & student profile management
- Expand recommendations to specific courses (link to Coursera/Udemy)

## Notes
- Keep your real dataset in `data/student_placement_dataset.csv` and ensure columns like `Placed`, `Salary`, `JobRole` exist or adapt the scripts.
