import argparse
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

TARGET = "price_in_inr"
NUM_FEATS = ["area_sqft", "bhk", "bathrooms", "floor", "total_floors"]
CAT_FEATS = ["locality", "property_type", "furnishing", "age_bucket"]

def build_preprocessor():
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    # sklearn 1.2+ uses sparse_output; older uses sparse
    try:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    except TypeError:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATS),
        ("cat", cat_pipe, CAT_FEATS)
    ])

def make_model(kind: str) -> Pipeline:
    if kind == "linear":
        reg = LinearRegression()
    elif kind == "rf":
        reg = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )
    else:
        raise ValueError("Unknown model")
    return Pipeline([("prep", build_preprocessor()), ("reg", reg)])

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/jalgaon_listings.csv")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--model", type=str, default="rf", choices=["linear", "rf"])
    args = ap.parse_args()

    out = Path(args.outdir); figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    needed = set(NUM_FEATS + CAT_FEATS + [TARGET])
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Your CSV is missing columns: {sorted(missing)}")

    X = df[NUM_FEATS + CAT_FEATS]
    y = df[TARGET]

    test_size = 0.2
    if len(df) < 10:
        test_size = max(2, int(round(0.2 * len(df))))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    model = make_model(args.model)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    r2, rmse, mae = metrics(yte, yhat)
    out.joinpath("metrics.txt").write_text(
        f"R2={r2:.4f}\nRMSE={rmse:.0f}\nMAE={mae:.0f}\n", encoding="utf-8"
    )
    joblib.dump(model, out / "model.pkl")

    # Parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(yte, yhat, alpha=0.7)
    lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
    plt.plot(lims, lims)
    plt.xlabel("True price")
    plt.ylabel("Predicted price")
    plt.title("Parity plot")
    plt.tight_layout()
    plt.savefig(figs / "parity.png", dpi=150); plt.close()

    # Residuals
    res = yte - yhat
    plt.figure(figsize=(6, 4))
    plt.scatter(yhat, res, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(figs / "residuals.png", dpi=150); plt.close()

    # 5-fold CV
    if len(df) >= 10:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        with open(out / "metrics.txt", "a", encoding="utf-8") as f:
            f.write(f"CV_R2_mean={cv_scores.mean():.4f}\nCV_R2_std={cv_scores.std():.4f}\n")
    else:
        with open(out / "metrics.txt", "a", encoding="utf-8") as f:
            f.write("CV skipped due to small dataset\n")

    # Permutation importance for RF on transformed features with readable names
    if args.model == "rf":
        prep = model.named_steps["prep"]
        reg = model.named_steps["reg"]

        # transform test features to the model input space
        Xte_t = prep.transform(Xte)

        # get expanded feature names
        try:
            feat_names = prep.get_feature_names_out()
        except AttributeError:
            num_names = list(NUM_FEATS)
            oh = prep.named_transformers_["cat"].named_steps["oh"]
            cat_names = oh.get_feature_names_out(CAT_FEATS).tolist()
            feat_names = num_names + cat_names

        pi = permutation_importance(
            reg, Xte_t, yte, n_repeats=5, random_state=42, n_jobs=-1
        )
        imp = pi.importances_mean

        if len(feat_names) != len(imp):
            raise SystemExit(
                f"Name/importance length mismatch: {len(feat_names)} vs {len(imp)}"
            )

        pd.DataFrame({"feature": feat_names, "importance": imp}) \
          .sort_values("importance", ascending=False) \
          .to_csv(out / "feature_importance.csv", index=False)

    print(f"Saved model to {out/'model.pkl'}")
    print(f"Saved metrics to {out/'metrics.txt'}")
    print(f"Saved figures to {figs}")
