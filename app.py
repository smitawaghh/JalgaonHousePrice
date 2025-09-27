import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

DEFAULT_MODEL = Path("outputs/model.pkl")
COLUMNS = ["area_sqft", "bhk", "bathrooms", "locality", "property_type",
           "furnishing", "age_bucket", "floor", "total_floors"]

LOCALITIES = [
    "Ring Road", "Navi Peth", "Khadke Nagar", "Mehrun", "Chandwad Naka",
    "RTO Office Area", "Ganesh Colony", "Jilha Peth", "Shiv Colony",
    "Bhagur", "Khedi", "Pimprala", "Tambapura", "MJ College Road", "Old Jalgaon"
]
PTYPES = ["Apartment", "Independent House", "Row House"]
FURNS = ["Unfurnished", "Semi Furnished", "Fully Furnished"]
AGES = ["0-5 yrs", "5-10 yrs", "10+ yrs"]

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

def predict_price(model, row):
    df = pd.DataFrame([row], columns=COLUMNS)
    return int(round(model.predict(df)[0]))

# Streamlit UI
st.set_page_config(page_title="Jalgaon House Price", page_icon="🏠", layout="centered")
st.title("🏠 Jalgaon House Price Predictor")
st.caption("Linear Regression and Random Forest with a clean preprocessing pipeline")

model_path = st.sidebar.text_input("Model path", value=str(DEFAULT_MODEL))
try:
    model = load_model(Path(model_path))
    st.sidebar.success("Model loaded")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    area = st.number_input("Area (sqft)", 300, 4000, 950, 10)
    bhk = st.number_input("BHK", 1, 6, 2, 1)
    bath = st.number_input("Bathrooms", 1, 6, 2, 1)
    floor = st.number_input("Floor", 1, 50, 3, 1)
with c2:
    tfloors = st.number_input("Total floors", 1, 60, 8, 1)
    locality = st.selectbox("Locality", LOCALITIES, index=LOCALITIES.index("Ring Road"))
    ptype = st.selectbox("Property type", PTYPES, index=0)
    furn = st.selectbox("Furnishing", FURNS, index=1)
    age = st.selectbox("Age bucket", AGES, index=0)

if st.button("Predict price"):
    row = {
        "area_sqft": area,
        "bhk": bhk,
        "bathrooms": bath,
        "locality": locality,
        "property_type": ptype,
        "furnishing": furn,
        "age_bucket": age,
        "floor": floor,
        "total_floors": tfloors
    }
    try:
        st.success(f"Predicted price: ₹ {predict_price(model, row):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("Feature importance (RF)"):
    p = Path("outputs/feature_importance.csv")
    if p.exists():
        st.write(pd.read_csv(p).head(15))
    else:
        st.write("Train with --model rf to generate this file.")

