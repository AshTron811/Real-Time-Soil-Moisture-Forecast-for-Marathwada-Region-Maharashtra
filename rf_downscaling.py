import ee
import streamlit as st
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import tempfile, os, json

# 1️⃣ EE init
def init_ee():
    raw = st.secrets["ee_credentials"]["json"]
    key_data = json.dumps(raw) if isinstance(raw, dict) else raw
    creds = json.loads(key_data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(key_data)
        key_path = tf.name
    try:
        ee.Initialize(ee.ServiceAccountCredentials(creds["client_email"], key_path))
    finally:
        os.remove(key_path)

# 2️⃣ Parameters
SMAP_COLL    = "NASA/SMAP/SPL4SMGP/007"
SMAP_BAND    = "sm_surface"
SMAP_SCALE   = 1000
RF_TREES     = 50
RF_MIN_SAMP  = 3
RF_MAX_NODES = 5
WINDOW_MO    = 12
BUFFER_M     = 5000

# 3️⃣ Build static predictor stack: Elev, Slope, 12-mo mean NDVI
def get_static_stack():
    elev  = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elev")
    slope = ee.Terrain.slope(elev).rename("slope")
    ndvi  = (ee.ImageCollection("MODIS/061/MOD13Q1")
               .filterDate(*date_window())
               .select("NDVI")
               .map(lambda i: i.multiply(0.0001))
               .mean()
               .rename("ndvi"))
    return elev.addBands(slope).addBands(ndvi)

# Date window helper
def date_window():
    end   = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.today() - relativedelta(months=WINDOW_MO)).strftime("%Y-%m-%d")
    return start, end

# 4️⃣ Load GAUL districts
def get_districts(names):
    gaul = (ee.FeatureCollection("FAO/GAUL/2015/level2")
             .filter(ee.Filter.eq("ADM0_NAME","India"))
             .filter(ee.Filter.eq("ADM1_NAME","Maharashtra")))
    return {d: gaul.filter(ee.Filter.eq("ADM2_NAME", d)).geometry().buffer(BUFFER_M)
            for d in names}

# 5️⃣ Downscaling loop
def downscale(district_geoms):
    static = get_static_stack()
    smap_ic = (ee.ImageCollection(SMAP_COLL)
                 .filterDate(*date_window())
                 .select(SMAP_BAND))
    times   = smap_ic.aggregate_array("system:time_start").getInfo()
    records = []

    prog = st.progress(0)
    for idx, ts in enumerate(times):
        date_str = ee.Date(ts).format("YYYY-MM-dd").getInfo()
        img       = ee.Image(smap_ic.filter(ee.Filter.eq("system:time_start", ts)).first())
        label     = img.select(SMAP_BAND).multiply(1000).round().rename("sm_int")
        stack     = static.addBands(label)

        for name, geom in district_geoms.items():
            # sample training pixels inside district
            samp = stack.sample(region=geom, scale=SMAP_SCALE, numPixels=2000, seed=42)
            rf   = (ee.Classifier.smileRandomForest(RF_TREES, RF_MIN_SAMP, RF_MAX_NODES)
                      .train(features=samp, classProperty="sm_int",
                             inputProperties=static.bandNames()))
            pred = static.classify(rf).divide(1000).rename("sm500m")

            # district-mean downscaled SM
            mean_sm = pred.reduceRegion(
                         ee.Reducer.mean(), geometry=geom,
                         scale=500, bestEffort=True
                      ).get("sm500m").getInfo()

            records.append({
                "district": name,
                "date":     date_str,
                "sm500m":   mean_sm
            })

        prog.progress((idx+1)/len(times))

    return pd.DataFrame.from_records(records)

# 6️⃣ Streamlit UI
st.title("RF Downscaling via GAUL Districts")
init_ee()
DIST_NAMES = ['Aurangabad','Bid','Hingoli','Jalna','Latur','Osmanabad','Parbhani','Nanded']
districts  = get_districts(DIST_NAMES)

st.write(f"Processing {len(DIST_NAMES)} districts over last {WINDOW_MO} months…")
df_series = downscale(districts)

st.success("Downscaling complete!")
st.dataframe(df_series)
csv = df_series.to_csv(index=False).encode("utf-8")
st.download_button("Download results as CSV", data=csv, file_name="sm_downscaled.csv")
