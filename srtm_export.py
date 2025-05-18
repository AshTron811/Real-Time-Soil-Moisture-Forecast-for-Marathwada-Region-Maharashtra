# srtm_export.py

import ee
import streamlit as st
import json, tempfile, os
import pandas as pd

st.title("1️⃣ SRTM Export (per-district, all pixels)")

def init_ee():
    if "ee_credentials" not in st.secrets:
        st.error("Missing ee_credentials"); st.stop()
    raw = st.secrets["ee_credentials"]["json"]
    key_data = json.dumps(raw) if isinstance(raw, dict) else raw.strip()
    creds = json.loads(key_data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(key_data); key_path = tf.name
    try:
        ee.Initialize(ee.ServiceAccountCredentials(creds["client_email"], key_path))
        st.success("✅ EE initialized")
    finally:
        os.remove(key_path)

def main():
    init_ee()
    DISTRICTS = ['Aurangabad','Bid','Hingoli','Jalna','Latur','Osmanabad','Parbhani','Nanded']
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")\
             .filter(ee.Filter.eq('ADM0_NAME','India'))\
             .filter(ee.Filter.eq('ADM1_NAME','Maharashtra'))

    elev  = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elev')
    slope = ee.Terrain.slope(elev).rename('slope')
    stack = elev.addBands(slope)

    all_rows = []
    header = None

    for d in DISTRICTS:
        st.info(f"Sampling SRTM in {d}…")
        region_i = gaul.filter(ee.Filter.eq('ADM2_NAME', d)).geometry().buffer(5000)
        arr = ee.ImageCollection([stack]).getRegion(region_i, 500).getInfo()
        if len(arr) < 2:
            st.warning(f"No pixels in {d}")
            continue
        if header is None:
            header = arr[0]
        all_rows += arr[1:]

    if not all_rows:
        st.error("No SRTM pixels found."); st.stop()

    df = pd.DataFrame(all_rows, columns=header)[['longitude','latitude','elev','slope']]
    df = df.rename(columns={'longitude':'lon','latitude':'lat'})
    df.to_csv('srtm_samples.csv', index=False)
    st.success(f"▶ srtm_samples.csv ({len(df)} rows)")  
    st.dataframe(df.head())

if __name__=='__main__':
    main()
