import base64
import datetime
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def load_workflow_spec():
    """Loads and unwraps the workflow JSON specification from the local app directory."""
    json_path = os.path.join(os.path.dirname(__file__), "workflow.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Could not find 'workflow.json' in the application directory. "
            "Please ensure your workflow.json file is placed in the same folder as app.py."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "specification" in data and isinstance(data["specification"], dict):
        return data["specification"]
    return data


def get_image_input_name(workflow_spec):
    """Inspects workflow_spec to find the expected image input variable name."""
    if isinstance(workflow_spec, dict) and "inputs" in workflow_spec:
        inputs = workflow_spec["inputs"]
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, dict):
                    inp_type = str(inp.get("type", "")).lower()
                    if "image" in inp_type or inp.get("name") in ["image", "images", "input_image"]:
                        return inp.get("name", "image")
    return "image"


def clean_scalar(val):
    """Sanitizes nested dicts/lists into clean scalar strings."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    val_str = str(val)
    if len(val_str) > 1000 or val_str.startswith("data:image"):
        return "[IMAGE/LONG BLOB DATA]"
    return val_str


def parse_workflow_outputs(data):
    """Safely parses outputs regardless of dict or list returned by Roboflow."""
    extracted = {}
    item = data[0] if isinstance(data, list) and len(data) > 0 else data

    if isinstance(item, dict):
        raw_outputs = item.get("outputs", item)
        if isinstance(raw_outputs, dict):
            for k, v in raw_outputs.items():
                extracted[k] = clean_scalar(v)
        elif isinstance(raw_outputs, list):
            for idx, elem in enumerate(raw_outputs):
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        key_name = f"{k}_{idx}" if k in extracted else k
                        extracted[key_name] = clean_scalar(v)
                else:
                    extracted[f"output_{idx}"] = clean_scalar(elem)
    elif isinstance(item, list):
        for idx, elem in enumerate(item):
            if isinstance(elem, dict):
                for k, v in elem.items():
                    key_name = f"{k}_{idx}" if k in extracted else k
                    extracted[key_name] = clean_scalar(v)
            else:
                extracted[f"output_{idx}"] = clean_scalar(elem)
                
    return extracted


def extract_images_from_uploads(uploaded_files):
    """Extracts raw image bytes while preserving folder structures and unpacking ZIPs."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    extracted_images = []

    for file_item in uploaded_files:
        filename = file_item.name
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".zip":
            try:
                with zipfile.ZipFile(file_item) as z:
                    for zip_info in z.infolist():
                        if not zip_info.is_dir() and not zip_info.filename.startswith("__MACOSX"):
                            zip_ext = os.path.splitext(zip_info.filename)[1].lower()
                            if zip_ext in valid_extensions:
                                img_bytes = z.read(zip_info.filename)
                                clean_name = os.path.basename(zip_info.filename)
                                if clean_name:
                                    extracted_images.append((clean_name, img_bytes))
            except Exception as e:
                st.error(f"Error reading ZIP file '{filename}': {e}")
        elif ext in valid_extensions:
            extracted_images.append((filename, file_item.getvalue()))

    return extracted_images


def run_one_image(filename, image_bytes, workflow_spec, user_api_key, image_input_name):
    """Executes inline workflow execution via Roboflow API."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://detect.roboflow.com/infer/workflows?api_key={user_api_key}"
    
    payload = {
        "specification": workflow_spec,
        "inputs": {
            image_input_name: {"type": "base64", "value": b64_image}
        },
        "api_key": user_api_key
    }
    
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if not response.ok:
            try:
                err_detail = response.json()
            except Exception:
                err_detail = response.text
            return {
                "Filename": filename,
                "Status": "Failed",
                "Date": date_str,
                "Timestamp": timestamp_str,
                "Error": f"HTTP {response.status_code}: {err_detail}"
            }

        data = response.json()
        
        row = {
            "Filename": filename, 
            "Status": "Success",
            "Date": date_str,
            "Timestamp": timestamp_str
        }
        
        parsed_outputs = parse_workflow_outputs(data)
        row.update(parsed_outputs)

        return row

    except Exception as e:
        return {
            "Filename": filename,
            "Status": "Failed",
            "Date": date_str,
            "Timestamp": timestamp_str,
            "Error": str(e)
        }


def build_daily_summary(df_results):
    """
    Groups detailed results and constructs the exact 15-column Daily Summary table
    with logic for summary_registration_source (CONFIRMED_DETECTION, 
    BEST_VISIBLE_CANDIDATE, or UNDETECTED).
    """
    summary_cols = [
        "capture_date",
        "summary_registration",
        "summary_registration_source",
        "appearance_count",
        "first_seen",
        "last_seen",
        "trip_length_hhmmss",
        "trip_length_seconds",
        "trip_length_status",
        "confirmed_registration_count",
        "partial_or_review_count",
        "undetected_count",
        "first_image",
        "last_image",
        "image_names"
    ]

    if df_results.empty:
        return pd.DataFrame(columns=summary_cols)

    df_work = df_results.copy()

    # Locate registration column dynamically
    reg_col = None
    for candidate in ["summary_registration", "registration_number", "registration", "ocr_text", "text"]:
        if candidate in df_work.columns:
            reg_col = candidate
            break
    if not reg_col:
        non_meta = [c for c in df_work.columns if c not in ["Filename", "Status", "Date", "Timestamp", "Error"]]
        reg_col = non_meta[0] if non_meta else "Registration"
        df_work[reg_col] = "UNDETECTED"

    # Locate status/confidence column dynamically
    status_col = None
    for candidate in ["detection_status", "review_status", "status_type", "registration_status", "status"]:
        if candidate in df_work.columns:
            status_col = candidate
            break

    # Standardize Timestamps & Date
    df_work["dt_ts"] = pd.to_datetime(df_work["Timestamp"], errors="coerce")
    df_work["capture_date"] = df_work["Date"]

    groups = df_work.groupby(["capture_date", reg_col])
    summary_rows = []

    for (cap_date, reg_val), group in groups:
        sorted_group = group.sort_values("dt_ts")
        
        appearance_count = len(sorted_group)
        first_dt = sorted_group["dt_ts"].min()
        last_dt = sorted_group["dt_ts"].max()
        
        first_seen = first_dt.strftime("%H:%M:%S") if pd.notnull(first_dt) else ""
        last_seen = last_dt.strftime("%H:%M:%S") if pd.notnull(last_dt) else ""
        
        # Calculate Trip Duration
        if pd.notnull(first_dt) and pd.notnull(last_dt):
            delta_sec = int((last_dt - first_dt).total_seconds())
            hours, remainder = divmod(delta_sec, 3600)
            minutes, seconds = divmod(remainder, 60)
            trip_hhmmss = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            delta_sec = 0
            trip_hhmmss = "00:00:00"

        trip_status = "COMPLETED" if appearance_count > 1 else "SINGLE_DETECTION"

        # Categorize detection status counts
        if status_col and status_col in sorted_group.columns:
            s_series = sorted_group[status_col].astype(str).str.upper()
            confirmed_cnt = (s_series == "CONFIRMED").sum()
            review_cnt = s_series.str.contains("REVIEW|PARTIAL|CANDIDATE", regex=True).sum()
            undetected_cnt = s_series.str.contains("UNDETECTED|NONE|FAILED", regex=True).sum()
        else:
            reg_str = str(reg_val).strip().upper()
            if reg_str in ["UNDETECTED", "NONE", "UNKNOWN", "", "NAN", "NULL"]:
                confirmed_cnt = 0
                review_cnt = 0
                undetected_cnt = appearance_count
            else:
                confirmed_cnt = appearance_count
                review_cnt = 0
                undetected_cnt = 0

        # Determine 'summary_registration_source'
        reg_clean = str(reg_val).strip().upper()
        if reg_clean in ["UNDETECTED", "NONE", "UNKNOWN", "", "NAN", "NULL"] or undetected_cnt == appearance_count:
            source_val = "UNDETECTED"
        elif confirmed_cnt > 0:
            source_val = "CONFIRMED_DETECTION"
        else:
            source_val = "BEST_VISIBLE_CANDIDATE"

        first_img = sorted_group["Filename"].iloc[0] if not sorted_group.empty else ""
        last_img = sorted_group["Filename"].iloc[-1] if not sorted_group.empty else ""
        image_names_str = ", ".join(sorted_group["Filename"].tolist())

        summary_rows.append({
            "capture_date": cap_date,
            "summary_registration": reg_val,
            "summary_registration_source": source_val,
            "appearance_count": appearance_count,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "trip_length_hhmmss": trip_hhmmss,
            "trip_length_seconds": delta_sec,
            "trip_length_status": trip_status,
            "confirmed_registration_count": confirmed_cnt,
            "partial_or_review_count": review_cnt,
            "undetected_count": undetected_cnt,
            "first_image": first_img,
            "last_image": last_img,
            "image_names": image_names_str
        })

    return pd.DataFrame(summary_rows, columns=summary_cols)


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================

if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None
if "excel_results_bytes" not in st.session_state:
    st.session_state.excel_results_bytes = None
if "excel_summary_bytes" not in st.session_state:
    st.session_state.excel_summary_bytes = None


# ==========================================
# 3. STREAMLIT UI & CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Boat Registration & Classification System",
    page_icon="🛥️",
    layout="wide"
)

st.title("🛥️ Boat Registration & Classification System")
st.markdown("Upload individual images, dropped folders, or ZIP archives for automated analysis.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Roboflow API Key", 
        type="password",
        help="Enter your personal Roboflow Private API Key."
    )
    
    st.divider()
    max_workers = st.slider("Parallel Processing Threads", min_value=1, max_value=10, value=4)

# File / Folder Dropzone
st.subheader("📁 Input Files")
uploaded_items = st.file_uploader(
    "Drag and drop image files, folders, or ZIP archives here:",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "zip"],
    accept_multiple_files=True
)

server_output_dir = st.text_input("Optional Server Save Path (Leave blank to download via browser):")


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

if st.button("🚀 Process Batch", type="primary", disabled=not (uploaded_items and api_key)):
    st.session_state.run_complete = False
    
    with st.status("Initializing processing session...", expanded=True) as status:
        
        status.update(label="Loading workflow specification...", state="running")
        try:
            workflow_spec = load_workflow_spec()
            image_input_name = get_image_input_name(workflow_spec)
        except Exception as e:
            st.error(str(e))
            st.stop()

        status.update(label="Extracting image batch...", state="running")
        image_batch = extract_images_from_uploads(uploaded_items)
        st.write(f"Total images identified for processing: **{len(image_batch)}**")

        if not image_batch:
            st.error("No valid image files found in the upload.")
            st.stop()

        status.update(label="Processing images via Roboflow API...", state="running")
        progress_bar = st.progress(0)
        
        results = []
        completed_count = 0
        total_count = len(image_batch)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    run_one_image, 
                    fn, 
                    bytes_data, 
                    workflow_spec,
                    api_key.strip(),
                    image_input_name
                ): fn for fn, bytes_data in image_batch
            }

            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
                completed_count += 1
                progress_bar.progress(completed_count / total_count)

        status.update(label="Processing complete! Generating output files...", state="complete")

    # Generate Output DataFrames
    df_res = pd.DataFrame(results)
    df_sum = build_daily_summary(df_res)

    # Output 1: Detailed Classification Results Excel
    excel_res_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_res_buffer, engine='openpyxl') as writer:
        df_res.to_excel(writer, sheet_name='Detailed Results', index=False)
    excel_res_bytes = excel_res_buffer.getvalue()

    # Output 2: Daily Registration Summary Excel
    excel_sum_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_sum_buffer, engine='openpyxl') as writer:
        df_sum.to_excel(writer, sheet_name='Daily Summary', index=False)
    excel_sum_bytes = excel_sum_buffer.getvalue()

    # Optional Server File Persistence
    if server_output_dir:
        os.makedirs(server_output_dir, exist_ok=True)
        with open(os.path.join(server_output_dir, "detailed_classification_results.xlsx"), "wb") as f:
            f.write(excel_res_bytes)
        with open(os.path.join(server_output_dir, "daily_registration_summary.xlsx"), "wb") as f:
            f.write(excel_sum_bytes)
        st.success(f"Both Excel files saved to server at: `{server_output_dir}`")

    # Session State Updates
    st.session_state.df_results = df_res
    st.session_state.df_summary = df_sum
    st.session_state.excel_results_bytes = excel_res_bytes
    st.session_state.excel_summary_bytes = excel_sum_bytes
    st.session_state.run_complete = True


# ==========================================
# 5. PERSISTENT DISPLAY & DOWNLOAD SECTION
# ==========================================

if st.session_state.run_complete:
    st.divider()
    st.header("📊 Results & Generated Excel Files")

    tab1, tab2 = st.tabs(["📄 Detailed Classification Data", "📅 Daily Registration Summary"])

    with tab1:
        st.dataframe(st.session_state.df_results, use_container_width=True)

    with tab2:
        st.dataframe(st.session_state.df_summary, use_container_width=True)

    st.subheader("📥 Download Output Excel Files")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📘 Download Detailed Results (.xlsx)",
            data=st.session_state.excel_results_bytes,
            file_name="detailed_classification_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label="📊 Download Daily Registration Summary (.xlsx)",
            data=st.session_state.excel_summary_bytes,
            file_name="daily_registration_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
