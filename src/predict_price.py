import argparse
import joblib
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="outputs/model.pkl")
    ap.add_argument("--area", type=float, required=True)
    ap.add_argument("--bhk", type=int, required=True)
    ap.add_argument("--bath", type=int, required=True)
    ap.add_argument("--loc", type=str, required=True)
    ap.add_argument("--ptype", type=str, required=True)
    ap.add_argument("--furn", type=str, required=True)
    ap.add_argument("--age", type=str, required=True)
    ap.add_argument("--floor", type=int, required=True)
    ap.add_argument("--tfloors", type=int, required=True)
    args = ap.parse_args()

    model = joblib.load(args.model)
    row = {
        "area_sqft": args.area,
        "bhk": args.bhk,
        "bathrooms": args.bath,
        "locality": args.loc,
        "property_type": args.ptype,
        "furnishing": args.furn,
        "age_bucket": args.age,
        "floor": args.floor,
        "total_floors": args.tfloors
    }
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    print(f"Predicted price: INR {pred:,.0f}")
