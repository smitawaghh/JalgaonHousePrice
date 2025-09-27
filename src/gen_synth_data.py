import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["area_sqft","bhk","bathrooms","locality","property_type","furnishing","age_bucket","floor","total_floors","price_in_inr"]

LOCALITIES = [
    "Ring Road","Navi Peth","Khadke Nagar","Mehrun","Chandwad Naka","RTO Office Area",
    "Ganesh Colony","Jilha Peth","Shiv Colony","Bhagur","Khedi","Pimprala","Tambapura",
    "MJ College Road","Old Jalgaon"
]
PTYPES = ["Apartment","Independent House","Row House"]
FURNS = ["Unfurnished","Semi Furnished","Fully Furnished"]
AGES = ["0-5 yrs","5-10 yrs","10+ yrs"]

# rough locality multipliers - tune later if you get real data
LOC_MUL = {
    "Ring Road":1.20,"Navi Peth":1.18,"Khadke Nagar":1.05,"Mehrun":1.07,"Chandwad Naka":1.02,
    "RTO Office Area":1.06,"Ganesh Colony":1.15,"Jilha Peth":1.03,"Shiv Colony":1.08,"Bhagur":0.95,
    "Khedi":0.92,"Pimprala":0.97,"Tambapura":0.96,"MJ College Road":1.10,"Old Jalgaon":0.98
}
PTYPE_MUL = {"Apartment":1.00,"Independent House":1.20,"Row House":1.10}
FURN_MUL = {"Unfurnished":0.98,"Semi Furnished":1.00,"Fully Furnished":1.05}
AGE_MUL = {"0-5 yrs":1.05,"5-10 yrs":1.00,"10+ yrs":0.95}

def sample_row(rng: np.random.Generator):
    # area drives bhk and price
    area = int(rng.normal(1000, 250))
    area = max(450, min(area, 2500))

    # bhk from area
    if area < 650:
        bhk = 1
    elif area < 1000:
        bhk = 2
    elif area < 1400:
        bhk = 3
    else:
        bhk = rng.integers(3, 5)  # 3 or 4

    # bathrooms close to bhk, min 1
    bathrooms = int(max(1, min(4, bhk + rng.integers(-1, 2))))

    locality = rng.choice(LOCALITIES).item()
    ptype = rng.choice(PTYPES).item()
    furnish = rng.choice(FURNS, p=[0.45, 0.35, 0.20]).item()
    age_bucket = rng.choice(AGES, p=[0.45, 0.35, 0.20]).item()

    # floors
    if ptype == "Independent House":
        total_floors = rng.integers(1, 3)
        floor = 1
    else:
        total_floors = int(rng.integers(4, 16))
        floor = int(rng.integers(1, total_floors + 1))

    # base price per sqft in INR - loose city-wide average
    base_pps = rng.normal(4200, 400)
    base_pps = max(3000, min(base_pps, 6000))

    # multipliers
    m = (
        LOC_MUL[locality]
        * PTYPE_MUL[ptype]
        * FURN_MUL[furnish]
        * AGE_MUL[age_bucket]
        * (1.00 + 0.01 * (floor - 1) if ptype != "Independent House" else 1.00)  # small floor premium
    )

    noise = rng.normal(1.0, 0.05)  # 5% noise
    price = int(round(area * base_pps * m * noise, -3))  # round to nearest 1k

    return {
        "area_sqft": area,
        "bhk": bhk,
        "bathrooms": bathrooms,
        "locality": locality,
        "property_type": ptype,
        "furnishing": furnish,
        "age_bucket": age_bucket,
        "floor": floor,
        "total_floors": total_floors,
        "price_in_inr": price
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of synthetic rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/jalgaon_listings_synth.csv")
    ap.add_argument("--append", action="store_true", help="append to existing jalgaon_listings.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [sample_row(rng) for _ in range(args.n)]
    df = pd.DataFrame(rows, columns=COLUMNS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    if args.append:
        base = Path("data/jalgaon_listings.csv")
        if base.exists():
            # append under header
            df.to_csv(base, mode="a", index=False, header=False)
            print(f"Appended {len(df)} rows to {base}")
        else:
            # create with header
            df.to_csv(base, index=False)
            print(f"Created {base} with {len(df)} rows")

    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
