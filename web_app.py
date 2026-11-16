"""
Streamlit Web Dashboard for Automatic Number Plate Recognition (ANPR)
Smart Parking Management Demonstration & Interactive Inspection
"""

import os
import sys
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import ANPRPipeline
from src.parking_system import SmartParkingSystem

st.set_page_config(
    page_title="ANPR Smart Parking System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
    .receipt-box {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 8px;
        border: 1px dashed #64748B;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(model_type="svm"):
    return ANPRPipeline(model_type=model_type)

def main():
    st.markdown('<p class="main-title">🚗 Intelligent ANPR Smart Parking System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pattern Recognition Project: Computer Vision, HOG-SVM Character Classification & Automated Gate Billing</p>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ System Configuration")
    model_choice = st.sidebar.selectbox(
        "Select OCR Pattern Classifier",
        ["svm", "mlp", "knn", "template"],
        index=0,
        help="SVM uses HOG descriptors with RBF kernel; MLP uses neural feedforward; k-NN uses instance matching; Template uses cross-correlation."
    )

    pipeline = load_pipeline(model_choice)
    parking = pipeline.parking

    # Real-time Occupancy Metrics
    occupancy = parking.get_parking_occupancy()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🅿️ Parking Lot Occupancy")
    col_o1, col_o2 = st.sidebar.columns(2)
    col_o1.metric("Available", occupancy["available"])
    col_o2.metric("Occupied", occupancy["occupied"])
    st.sidebar.progress(occupancy["occupied"] / max(1, occupancy["total"]))

    # Image Input Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📷 Vehicle Image Source")
    input_mode = st.sidebar.radio("Select Source", ["Benchmark Sample", "Upload Custom Image"])

    image_to_process = None
    sample_name = ""

    raw_dir = "data/raw"
    sample_files = []
    if os.path.exists(raw_dir):
        sample_files = [f for f in os.listdir(raw_dir) if f.endswith(".png")]

    if input_mode == "Benchmark Sample" and sample_files:
        sample_name = st.sidebar.selectbox("Select Surveillance Frame", sorted(sample_files))
        image_to_process = os.path.join(raw_dir, sample_name)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_to_process = temp_path

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["🚦 Live Gate Operations", "🔬 Pattern Recognition Pipeline Inspector", "📊 Performance Analytics & Database"])

    with tab1:
        st.subheader("Smart Barrier Gate Simulation")
        col_gate1, col_gate2 = st.columns([1, 1])

        with col_gate1:
            gate_action = st.radio("Select Gate Action", ["Vehicle Entry Gate", "Vehicle Exit Gate"], horizontal=True)
            mall_spent = 0.0
            duration_mins = 120.0
            if gate_action == "Vehicle Exit Gate":
                c_m1, c_m2 = st.columns(2)
                duration_mins = c_m1.slider("Parking Duration (Mins)", 10, 480, 140, step=10)
                mall_spent = c_m2.number_input("Mall Shopping Bill (₹)", min_value=0.0, value=650.0, step=50.0)

            run_btn = st.button("🚀 Process Surveillance Camera Feed", type="primary", use_container_width=True)

        if image_to_process and run_btn:
            action_code = "entry" if gate_action == "Vehicle Entry Gate" else "exit"
            with st.spinner("Executing Pattern Recognition Pipeline..."):
                res = pipeline.process(
                    image_to_process,
                    action=action_code,
                    mall_spent=mall_spent,
                    simulated_duration_mins=duration_mins
                )

            with col_gate2:
                st.image(cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB), caption="Surveillance Camera Feed (Annotated Detection)", use_container_width=True)

            st.markdown("---")
            # Result Presentation
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Recognized Plate", res["final_plate"])
            r2.metric("OCR Confidence", f"{res['ocr_confidence']*100:.1f}%")
            r3.metric("Tilt Correction", f"{res['detection']['angle']:.1f}°")
            r4.metric("Plate Aspect Ratio", f"{res['detection']['bbox'][2]/max(1, res['detection']['bbox'][3]):.2f}")

            # Smart Parking Gate Response
            gate_res = res["parking_result"]
            if gate_res:
                if gate_res.get("success"):
                    st.success(f"✅ Barrier Action: **{gate_res['barrier']}** — {gate_res['message']}")
                    
                    if action_code == "entry":
                        st.info(f"🎫 **Issued Entry Ticket** | Plate: **{gate_res['plate_number']}** | Permittee: **{gate_res['owner_name']} ({gate_res['category']})** | Assigned Slot: **{gate_res['slot_id']}**")
                    else:
                        st.markdown(f"""
                        <div class="receipt-box">
                        =======================================================<br>
                                   METROPOLIS MALL SMART PARKING RECEIPT<br>
                        =======================================================<br>
                        Receipt ID       : REC-{gate_res['log_id']:05d}<br>
                        Vehicle Number   : {gate_res['plate_number']}<br>
                        Category         : {gate_res['category']}<br>
                        Parked Slot      : {gate_res['slot_id']}<br>
                        Entry Time       : {gate_res['entry_time']}<br>
                        Exit Time        : {gate_res['exit_time']}<br>
                        Duration Parked  : {gate_res['duration_formatted']} ({gate_res['duration_minutes']} mins)<br>
                        -------------------------------------------------------<br>
                        Base Parking Fee : ₹{gate_res['base_fee']:.2f}<br>
                        Mall Discount    : -₹{gate_res['discount_amount']:.2f}<br>
                        <b>TOTAL PAID       : ₹{gate_res['final_amount']:.2f}</b><br>
                        Payment Method   : {gate_res['payment_method']} (Electronic Auto-Debit)<br>
                        Payment Status   : {gate_res['payment_status']}<br>
                        Exit Barrier     : [{gate_res['barrier']}]<br>
                        =======================================================<br>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ {gate_res.get('message')}")

    with tab2:
        st.subheader("Step-by-Step Computer Vision & Pattern Recognition Stages")
        if image_to_process:
            res_dbg = pipeline.process(image_to_process, action="detect_only")
            
            s1, s2, s3 = st.columns(3)
            s1.image(cv2.cvtColor(res_dbg["original_image"], cv2.COLOR_BGR2RGB), caption="1. Original Surveillance Frame")
            s2.image(res_dbg["preprocessing"]["clahe"], caption="2. CLAHE Local Contrast Enhanced", clamp=True)
            s3.image(res_dbg["preprocessing"]["candidate_mask"], caption="3. Morphological Vertical Candidate Mask", clamp=True)

            s4, s5, s6 = st.columns(3)
            s4.image(cv2.cvtColor(res_dbg["plate_crop"], cv2.COLOR_BGR2RGB), caption=f"4. Deskewed License Plate Crop ({res_dbg['detection']['angle']:.1f}°)")
            s5.image(res_dbg["seg_debug"]["binary"], caption="5. Binarized Plate (Adaptive Gaussian)", clamp=True)
            
            # Show Segmented Characters
            st.markdown("##### 6. Segmented Alphanumeric Glyphs & Class Predictions")
            if res_dbg["char_details"]:
                char_cols = st.columns(min(12, len(res_dbg["char_details"])))
                for idx, cinfo in enumerate(res_dbg["char_details"]):
                    with char_cols[idx % len(char_cols)]:
                        char_crop = res_dbg["segmentation"][idx]["char_img"]
                        st.image(char_crop, width=50, clamp=True)
                        st.caption(f"**{cinfo['char']}**\n({cinfo['confidence']*100:.0f}%)")

            st.info(f"**Post-Processing Resolution:** Raw OCR `'{res_dbg['raw_text']}'` ➔ Syntactic Disambiguation `'{res_dbg['disambiguated_text']}'` ➔ Verified Match `'{res_dbg['final_plate']}'`")

    with tab3:
        st.subheader("System Benchmarking & Relational Database")
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown("##### OCR Model Comparison")
            if os.path.exists("outputs/ocr_benchmark_metrics.csv"):
                df_ocr = pd.read_csv("outputs/ocr_benchmark_metrics.csv")
                st.dataframe(df_ocr, use_container_width=True)
            if os.path.exists("outputs/figures/model_comparison_bar.png"):
                st.image("outputs/figures/model_comparison_bar.png", caption="Model Performance Breakdown")

        with col_b2:
            st.markdown("##### Environmental Robustness Across Conditions")
            if os.path.exists("outputs/environmental_benchmark.csv"):
                df_env = pd.read_csv("outputs/environmental_benchmark.csv")
                st.dataframe(df_env, use_container_width=True)
            if os.path.exists("outputs/figures/environmental_robustness.png"):
                st.image("outputs/figures/environmental_robustness.png", caption="Environmental Performance Breakdown")

        st.markdown("---")
        st.subheader("Live SQLite Database Records")
        conn = parking.get_connection()
        t_slots, t_reg, t_logs = st.tabs(["Parking Slots", "Registered Permittees", "Parking Activity Logs"])
        with t_slots:
            st.dataframe(pd.read_sql_query("SELECT * FROM parking_slots LIMIT 25", conn), use_container_width=True)
        with t_reg:
            st.dataframe(pd.read_sql_query("SELECT * FROM registered_vehicles", conn), use_container_width=True)
        with t_logs:
            st.dataframe(pd.read_sql_query("SELECT * FROM parking_logs ORDER BY log_id DESC LIMIT 25", conn), use_container_width=True)
        conn.close()

if __name__ == "__main__":
    main()
