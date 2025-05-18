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
key_data = st.secrets["ee_credentials"]["json"]
if isinstance(key_data, dict):
    key_data = json.dumps(key_data)
creds = json.loads(key_data)
st.write("• Retrieved credentials from secrets.")
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write(key_data)
    path = f.name
try:
    ee.Initialize(ee.ServiceAccountCredentials(creds['client_email'], path))
    st.success("✅ Earth Engine initialized")
except Exception as e:
    st.error(f"EE init error: {e}")
    st.stop()
finally:
    if os.path.exists(path):
        os.remove(path)
        st.write("• Temporary credentials file removed.")

# ------------------ Auto-refresh at midnight ------------------
st.write("⏰ Configuring auto-refresh at next midnight...")
now = datetime.datetime.now()
midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min)
ms_until_midnight = int((midnight - now).total_seconds() * 1000)
st.write(f"• Milliseconds until midnight: {ms_until_midnight}")
st_autorefresh(interval=ms_until_midnight, limit=1, key="autoRefresh")

# ------------------ Hard-coded Districts ------------------
DISTRICTS = [
    'Aurangabad', 'Bid', 'Hingoli', 'Jalna',
    'Latur', 'Osmanabad', 'Parbhani', 'Nanded'
]

# ------------------ Load GAUL boundaries ------------------
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2") \
         .filter(ee.Filter.eq('ADM0_NAME', 'India')) \
         .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'))

# ------------------ Date Range ------------------
today = datetime.date.today()
start_date = (today - relativedelta(months=12)).strftime("%Y-%m-%d")
end_date   = today.strftime("%Y-%m-%d")
st.write(f"**Data range:** {start_date} to {end_date}")

# ------------------ MODIS Collections & Bands ------------------
collections = {
    'LST_Emissivity':    ('MODIS/061/MOD11A1', ['LST_Day_1km','Emis_31','Emis_32']),
    'Vegetation':        ('MODIS/061/MOD13Q1', ['NDVI','EVI']),
    'ET':                ('MODIS/061/MOD16A2', ['ET']),
    'SurfaceReflectance':('MODIS/061/MOD09GA', ['sur_refl_b01','sur_refl_b02','sur_refl_b03']),
    'SnowCover':         ('MODIS/061/MOD10A1', ['NDSI_Snow_Cover'])
}
all_bands = [b for _,bl in collections.values() for b in bl]

@st.cache_data(ttl=ms_until_midnight/1000, show_spinner=True)
def get_modis_data(districts):
    records = []
    # Calculate total images across collections for progress
    total_images = sum(
        ee.ImageCollection(col_id)
          .filterDate(start_date, end_date)
          .size().getInfo()
        for col_id,_ in collections.values()
    )
    prog = st.progress(0)
    status = st.empty()
    img_counter = 0

    for name,(col_id,bands) in collections.items():
        status.write(f"🔄 Collection: {name}")
        col = ee.ImageCollection(col_id).filterDate(start_date, end_date)
        n_imgs = col.size().getInfo()
        imgs = col.toList(n_imgs)
        for i in range(n_imgs):
            img = ee.Image(imgs.get(i))
            date = img.date().format('YYYY-MM-dd').getInfo()
            status.write(f"  ▶ Image {i+1}/{n_imgs} (Date: {date})")

            for d in districts:
                geom = gaul.filter(ee.Filter.eq('ADM2_NAME', d)).geometry().buffer(5000)
                stats = img.select(bands).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=geom,
                    scale=1000, bestEffort=True
                ).getInfo()
                rec = {'date': date, 'product': name, 'district': d}
                for b in bands:
                    rec[b] = stats.get(b)
                records.append(rec)

            img_counter += 1
            prog.progress(img_counter / total_images)

    if not records:
        return pd.DataFrame(columns=['date','product','district']+all_bands)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['product','date','district'])

# ------------------ Run & Display ------------------
modis_df = get_modis_data(DISTRICTS)
st.write("### MODIS per-district time series")
st.dataframe(modis_df)

csv = modis_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name="modis_by_district.csv")