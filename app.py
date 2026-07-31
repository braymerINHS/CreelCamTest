import base64
import datetime
import io
import json
import os
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 1. HELPER & CLEANING FUNCTIONS
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


def clean_scalar(value):
    """
    Identical scalar cleaner from your standalone Python script.
    Converts workflow outputs to spreadsheet-safe values.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ").strip()

    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned = clean_scalar(item)
            if cleaned != "":
                cleaned_items.append(str(cleaned))
        return " | ".join(cleaned_items)

    if isinstance(value, dict):
        if value.get("type") == "base64":
            return "[base64 image omitted]"

        if "predictions" in value:
            preds = value.get("predictions") or []
            return f"[{len(preds)} predictions omitted]"

        preferred_keys = [
            "boat_type",
            "confidence",
            "reason",
            "approved_identifiers_found",
            "unapproved_observations_report",
            "output",
            "parsed_output",
            "result",
            "text",
        ]

        parts = []
        for key in preferred_keys:
            if key in value:
                cleaned = clean_scalar(value.get(key))
                if cleaned != "":
                    parts.append(f"{key}: {cleaned}")

        if parts:
            return " | ".join(parts)

        return "[dict omitted]"

    return str(value).replace("\r", " ").replace("\n", " ").strip()


def first_result(result):
    """Extracts the output dictionary from Roboflow response."""
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            return item.get("outputs", item)
        return {}
    if isinstance(result, dict):
        return result.get("outputs", result)
    return {}


def extract_images_from_uploads(uploaded_files):
    """Extracts raw image bytes preserving file names and unpacking ZIPs."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
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


def run_one_image(filename, image_bytes, workflow_spec, user_api_key, image_input_name, run_id, run_started_at):
    """Executes infer workflow and extracts exact 24 fields matching your script."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://detect.roboflow.com/infer/workflows?api_key={user_api_key}"
    
    payload = {
        "specification": workflow_spec,
        "inputs": {
            image_input_name: {"type": "base64", "value": b64_image}
        },
        "api_key": user_api_key
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if not response.ok:
            try:
                err_detail = response.json()
            except Exception:
                err_detail = response.text
            return {
                "run_id": run_id,
                "run_started_at": run_started_at,
                "source_image": filename,
                "file_name": filename,
                "status": "workflow_error",
                "registration_number": "",
                "partial_registration_candidate": "",
                "boat_count": "",
                "boat_detected": "",
                "date_and_time": "",
                "output_line": "",
                "raw_ocr_text": "",
                "all_visible_text": "",
                "other_visible_text": "",
                "easyocr_visible_text": "",
                "verified_text_region_ocr": "",
                "focused_registration_ocr": "",
                "boat_type": "",
                "boat_type_identifier_confidence": "",
                "approved_identifiers_found": "",
                "unapproved_identifiers": "",
                "classification_reason": "",
                "workflow_error": f"HTTP {response.status_code}: {err_detail}",
                "attempts": 1,
            }

        data = response.json()
        output = first_result(data)

        return {
            "run_id": run_id,
            "run_started_at": run_started_at,
            "source_image": filename,
            "file_name": filename,
            "status": clean_scalar(output.get("status")),
            "registration_number": clean_scalar(output.get("registration_number")),
            "partial_registration_candidate": clean_scalar(output.get("partial_registration_candidate")),
            "boat_count": clean_scalar(output.get("boat_count")),
            "boat_detected": clean_scalar(output.get("boat_detected")),
            "date_and_time": clean_scalar(output.get("date_and_time")),
            "output_line": clean_scalar(output.get("output_line")),
            "raw_ocr_text": clean_scalar(output.get("raw_ocr_text")),
            "all_visible_text": clean_scalar(output.get("all_visible_text")),
            "other_visible_text": clean_scalar(output.get("other_visible_text")),
            "easyocr_visible_text": clean_scalar(output.get("easyocr_visible_text")),
            "verified_text_region_ocr": clean_scalar(output.get("verified_text_region_ocr")),
            "focused_registration_ocr": clean_scalar(output.get("focused_registration_ocr")),
            "boat_type": clean_scalar(output.get("boat_type")),
            "boat_type_identifier_confidence": clean_scalar(output.get("boat_type_identifier_confidence")),
            "approved_identifiers_found": clean_scalar(output.get("approved_identifiers_found")),
            "unapproved_identifiers": clean_scalar(output.get("unapproved_identifiers")),
            "classification_reason": clean_scalar(output.get("classification_reason")),
            "workflow_error": "",
            "attempts": 1,
        }

    except Exception as e:
        return {
            "run_id": run_id,
            "run_started_at": run_started_at,
            "source_image": filename,
            "file_name": filename,
            "status": "script_error",
            "registration_number": "",
            "partial_registration_candidate": "",
            "boat_count": "",
            "boat_detected": "",
            "date_and_time": "",
            "output_line": "",
            "raw_ocr_text": "",
            "all_visible_text": "",
            "other_visible_text": "",
            "easyocr_visible_text": "",
            "verified_text_region_ocr": "",
            "focused_registration_ocr": "",
            "boat_type": "",
            "boat_type_identifier_confidence": "",
            "approved_identifiers_found": "",
            "unapproved_identifiers": "",
            "classification_reason": "",
            "workflow_error": str(e),
            "attempts": 1,
        }


def extract_date_only(date_and_time):
    """Extracts date string from date_and_time field."""
    text = str(date_and_time or "").strip()
    if not text:
        return ""
    return text.split()[0] if text.split() else ""


def build_daily_summary(rows, run_id, run_started_at):
    """
    Identical aggregation logic to write_daily_summary_csv in your script.
    Grouped by (date, registration_number) where registration_number is non-empty.
    """
    summary = defaultdict(lambda: {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "date": "",
        "registration_number": "",
        "image_count": 0,
        "images": [],
    })

    for row in rows:
        registration = str(row.get("registration_number") or "").strip()
        if not registration:
            continue

        date = extract_date_only(row.get("date_and_time"))
        key = (date, registration)

        summary[key]["date"] = date
        summary[key]["registration_number"] = registration
        summary[key]["image_count"] += 1
        summary[key]["images"].append(row.get("file_name", ""))

    summary_rows = []
    for (_, _), item in sorted(summary.items(), key=lambda x: (x[0][0], x[0][1])):
        summary_rows.append({
            "run_id": item["run_id"],
            "run_started_at": item["run_started_at"],
            "date": item["date"],
            "registration_number": item["registration_number"],
            "image_count": item["image_count"],
            "images": " | ".join(item["images"]),
        })

    summary_cols = [
        "run_id",
        "run_started_at",
        "date",
        "registration_number",
        "image_count",
        "images",
    ]

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
st.markdown("Upload individual images, dropped folders, or ZIP archives for automated batch processing.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Roboflow API Key", 
        type="password",
        help="Enter your personal Roboflow Private API Key."
    )
    
    st.divider()
    max_workers = st.slider("Parallel Processing Threads", min_value=1, max_value=10, value=3)

# File / Folder Dropzone
st.subheader("📁 Input Files")
uploaded_items = st.file_uploader(
    "Drag and drop image files, folders, or ZIP archives here:",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "zip"],
    accept_multiple_files=True
)

server_output_dir = st.text_input("Optional Server Save Path (Leave blank to download via browser):")


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

if st.button("🚀 Process Batch", type="primary", disabled=not (uploaded_items and api_key)):
    st.session_state.run_complete = False
    
    # Generate run ID and timestamp for this batch run
    run_started_at_dt = datetime.datetime.now()
    run_id = run_started_at_dt.strftime("%Y-%m-%d_%H-%M-%S")
    run_started_at = run_started_at_dt.isoformat(timespec="seconds")

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
                    image_input_name,
                    run_id,
                    run_started_at
                ): fn for fn, bytes_data in image_batch
            }

            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
                completed_count += 1
                progress_bar.progress(completed_count / total_count)

        status.update(label="Processing complete! Generating output files...", state="complete")

    # Define exact field ordering from your script for results CSV/Excel
    fieldnames = [
        "run_id",
        "run_started_at",
        "source_image",
        "file_name",
        "status",
        "registration_number",
        "partial_registration_candidate",
        "boat_count",
        "boat_detected",
        "date_and_time",
        "output_line",
        "raw_ocr_text",
        "all_visible_text",
        "other_visible_text",
        "easyocr_visible_text",
        "verified_text_region_ocr",
        "focused_registration_ocr",
        "boat_type",
        "boat_type_identifier_confidence",
        "approved_identifiers_found",
        "unapproved_identifiers",
        "classification_reason",
        "workflow_error",
        "attempts",
    ]

    # Sort results by source_image exactly as script does
    results = sorted(results, key=lambda r: r.get("source_image", ""))

    df_res = pd.DataFrame(results, columns=fieldnames)
    df_sum = build_daily_summary(results, run_id, run_started_at)

    # Output 1: Detailed Classification Results Excel
    excel_res_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_res_buffer, engine='openpyxl') as writer:
        df_res.to_excel(writer, sheet_name='Clean Results', index=False)
    excel_res_bytes = excel_res_buffer.getvalue()

    # Output 2: Daily Registration Summary Excel
    excel_sum_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_sum_buffer, engine='openpyxl') as writer:
        df_sum.to_excel(writer, sheet_name='Daily Summary', index=False)
    excel_sum_bytes = excel_sum_buffer.getvalue()

    # Optional Server File Persistence
    if server_output_dir:
        os.makedirs(server_output_dir, exist_ok=True)
        with open(os.path.join(server_output_dir, f"roboflow_results_clean_{run_id}.xlsx"), "wb") as f:
            f.write(excel_res_bytes)
        with open(os.path.join(server_output_dir, f"registration_daily_summary_{run_id}.xlsx"), "wb") as f:
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

    tab1, tab2 = st.tabs(["📄 Detailed Clean Results", "📅 Registration Daily Summary"])

    with tab1:
        st.dataframe(st.session_state.df_results, use_container_width=True)

    with tab2:
        st.dataframe(st.session_state.df_summary, use_container_width=True)

    st.subheader("📥 Download Output Excel Files")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📘 Download Clean Results (.xlsx)",
            data=st.session_state.excel_results_bytes,
            file_name="roboflow_results_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label="📊 Download Registration Daily Summary (.xlsx)",
            data=st.session_state.excel_summary_bytes,
            file_name="registration_daily_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
