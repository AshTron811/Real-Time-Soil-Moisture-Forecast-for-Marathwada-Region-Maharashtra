import ee
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import tempfile, os, json

# ------------------ Earth Engine Authentication ------------------
st.write("🔑 Starting Earth Engine authentication...")
if "ee_credentials" not in st.secrets:
    st.error("Missing 'ee_credentials' in secrets. Please add your service account JSON.")
    st.stop()
raw = st.secrets["ee_credentials"]["json"]
key_data = json.dumps(raw) if isinstance(raw, dict) else raw
creds = json.loads(key_data)
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write(key_data)
    path = f.name
try:
    ee.Initialize(ee.ServiceAccountCredentials(creds['client_email'], path))
    st.success("✅ Earth Engine initialized successfully.")
except Exception as e:
    st.error(f"Earth Engine initialization error: {e}")
    st.stop()
finally:
    os.remove(path)

# ------------------ Auto-refresh at midnight ------------------
now = datetime.datetime.now()
midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min)
ms_until_midnight = int((midnight - now).total_seconds() * 1000)
st.write(f"⏰ Auto-refresh in {ms_until_midnight//1000} seconds (at next midnight)")
st_autorefresh(interval=ms_until_midnight, limit=1, key="autoRefresh")

# ------------------ Hard-coded Districts ------------------
DISTRICTS = ['Aurangabad','Bid','Hingoli','Jalna','Latur','Osmanabad','Parbhani','Nanded']

# ------------------ Load GAUL boundaries ------------------
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2") \
    .filter(ee.Filter.eq('ADM0_NAME', 'India')) \
    .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'))

# ------------------ Date Range ------------------
today = datetime.date.today()
start_date = (today - relativedelta(months=12)).strftime("%Y-%m-%d")
end_date   = today.strftime("%Y-%m-%d")
st.write(f"**Data range:** {start_date} to {end_date}")

# ------------------ GLDAS Collection & Variables ------------------
collection_id = 'NASA/GLDAS/V021/NOAH/G025/T3H'
collections = {
    'NetShortwaveFlux':      ['Swnet_tavg'],
    'NetLongwaveFlux':       ['Lwnet_tavg'],
    'LatentHeatFlux':        ['Qle_tavg'],
    'SensibleHeatFlux':      ['Qh_tavg'],
    'Evapotranspiration':     ['Evap_tavg'],
    'AvgSurfaceSkinTemp':    ['AvgSurfT_inst'],
    'RootZoneSoilMoisture':  ['RootMoist_inst'],
    'Transpiration':         ['Tveg_tavg']
}
st.write(f"• Collection: {collection_id}")
st.write(f"• Variables: {list(collections.keys())}")

# ------------------ Fetch GLDAS Data ------------------
def get_gldas_data(districts):
    records = []
    col = ee.ImageCollection(collection_id).filterDate(start_date, end_date)
    total_images = col.size().getInfo()
    img_list = col.toList(total_images)

    # Loop over each image
    for idx in range(total_images):
        img = ee.Image(img_list.get(idx))
        date_str = img.date().format('YYYY-MM-dd').getInfo()
        st.write(f"🔄 Image {idx+1} of {total_images}: {date_str}")
        for d in districts:
            st.write(f"  • District {d}")
            geom = gaul.filter(ee.Filter.eq('ADM2_NAME', d)).geometry().buffer(5000)
            props = img.select(*[b for bands in collections.values() for b in bands]) \
                       .reduceRegion(ee.Reducer.mean(), geom, scale=27830, bestEffort=True) \
                       .getInfo() or {}
            for var, band_list in collections.items():
                records.append({
                    'date':     date_str,
                    'variable': var,
                    'district': d,
                    'value':    props.get(band_list[0])
                })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['district','variable','date'])

# ------------------ Main ------------------
st.write("🚀 Fetching GLDAS data by district...")
df = get_gldas_data(DISTRICTS)
st.write("✅ Fetch complete.")

st.write("### GLDAS Time Series by District")
st.dataframe(df)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name='gldas_by_district.csv')