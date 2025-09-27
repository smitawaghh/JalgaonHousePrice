# Jalgaon House Price Predictor

Predict house prices in **Jalgaon** using a simple, reliable scikit-learn pipeline. Fast to run. Easy to explain. Recruiter friendly.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![VS Code](https://img.shields.io/badge/Editor-VS%20Code-1f425f.svg)

---

## What this repo shows
- End to end price prediction for Jalgaon
- Two models: **Linear Regression** and **Random Forest**
- Proper preprocessing with `ColumnTransformer`
- Holdout metrics and 5 fold cross validation
- Parity and residual plots
- Permutation feature importance for Random Forest

---

## Quickstart

```bash
pip install -r requirements.txt

# train linear
python src/train_jalgaon.py --data data/jalgaon_listings.csv --outdir outputs_linear --model linear

# train random forest
python src/train_jalgaon.py --data data/jalgaon_listings.csv --outdir outputs --model rf

# predict one listing using the RF model
python src/predict_price.py --model outputs/model.pkl \
  --area 950 --bhk 2 --bath 2 --loc "Ring Road" \
  --ptype "Apartment" --furn "Semi Furnished" --age "0-5 yrs" \
  --floor 3 --tfloors 8
esults on sample data

The RF numbers below are from outputs/metrics.txt. Linear results are in outputs_linear/metrics.txt.

Model	R2	MAE (INR)	RMSE (INR)	CV R2 mean ± std
Random Forest	0.8127	488,246	633,807	0.7531 ± 0.0589
Linear	see outputs_linear/metrics.txt	