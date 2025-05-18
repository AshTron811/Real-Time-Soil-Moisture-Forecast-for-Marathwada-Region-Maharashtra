import ee
import streamlit as st
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import tempfile, os, json

# ------------------ Earth Engine Initialization ------------------
def init_ee():
    if "ee_credentials" not in st.secrets:
        st.error("Missing Earth Engine credentials in secrets.")
        st.stop()
    raw = st.secrets["ee_credentials"]["json"]
    key_data = json.dumps(raw) if isinstance(raw, dict) else raw
    creds = json.loads(key_data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(key_data)
        key_path = tf.name
    try:
        ee.Initialize(ee.ServiceAccountCredentials(creds['client_email'], key_path))
        st.success("✅ Earth Engine initialized")
    finally:
        os.remove(key_path)

# ------------------ Parameters ------------------
DISTRICTS = ['Aurangabad','Bid','Hingoli','Jalna','Latur','Osmanabad','Parbhani','Nanded']
BUFFER_M = 5000  # buffer around district boundaries
START = datetime.date.today() - relativedelta(months=12)
END   = datetime.date.today()

# ------------------ Load GAUL and Prepare District Polygons ------------------
init_ee()
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2") \
    .filter(ee.Filter.eq('ADM0_NAME','India')) \
    .filter(ee.Filter.eq('ADM1_NAME','Maharashtra'))

# Build a dict of district geometries
district_geoms = {d: gaul.filter(ee.Filter.eq('ADM2_NAME', d)).geometry().buffer(BUFFER_M) 
                  for d in DISTRICTS}

# ------------------ CHIRPS Precipitation Export ------------------
# Daily precipitation raster
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
    .filterDate(START.strftime('%Y-%m-%d'), END.strftime('%Y-%m-%d'))

st.title("🌧️ CHIRPS Precipitation Export by District")
st.write(f"Period: {START} to {END}")
progress = st.progress(0)
status = st.empty()
all_records = []

# Iterate per image and per district
dates = chirps.aggregate_array('system:time_start').getInfo()
total = len(dates)
for idx, ts in enumerate(dates):
    img = ee.Image(chirps.filter(ee.Filter.eq('system:time_start', ts)).first())
    date_str = img.date().format('YYYY-MM-dd').getInfo()
    status.write(f"Processing image {idx+1}/{total}: {date_str}")
    # Sample every pixel in each district
    for d, geom in district_geoms.items():
        pts = img.sample(
            region=geom,
            scale=5000,
            geometries=True
        ).getInfo()['features']
        for f in pts:
            coords = f['geometry']['coordinates']
            precip = f['properties']['precipitation']
            all_records.append({
                'district': d,
                'date': date_str,
                'lon': coords[0],
                'lat': coords[1],
                'precip_mm': precip
            })
    progress.progress((idx+1)/total)

# Build DataFrame and display
df = pd.DataFrame(all_records)
df['date'] = pd.to_datetime(df['date'])

st.write(f"Exported {len(df)} pixels")
st.dataframe(df)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download pixel-level CSV", data=csv, file_name='chirps_by_district_pixels.csv')