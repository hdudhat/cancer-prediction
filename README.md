# 🩺 Cancer Prediction — ML + Streamlit (Demo)

> Educational demo — not medical advice or a medical device.

An end-to-end binary classification project to estimate cancer presence from medical & lifestyle features.  
Includes data validation, feature engineering, model training (LogReg, RandomForest, GB), thresholding by specificity, and a Streamlit app.

## Project Structure

python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# Put your CSV at data/raw/cancer.csv

python -m scripts.prepare_data
python -m scripts.run_baselines
python -m scripts.build_artifact

cd app
streamlit run streamlit_app.py
