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
# CONSTANTS & MASTER TEMPLATE CONFIG
# ==========================================
SOURCE_WORKSPACE = "blakes-workspace-efak8"
DEFAULT_WORKFLOW = "boat-registration-and-classification-combination"


# ==========================================
# 1. HELPER FUNCTIONS & WORKFLOW SETUP
# ==========================================

def clean_scalar(val):
    """
    Sanitizes nested dicts, lists, and long binary string blobs into 
    clean, spreadsheet-friendly scalar strings.
    """
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    val_str = str(val)
    if len(val_str) > 1000 or val_str.startswith("data:image"):
        return "[IMAGE/LONG BLOB DATA]"
    return val_str


def ensure_workflow_exists(api_key, target_workspace, source_workspace, workflow_id):
    """
    Verifies if the workflow exists in the user's target workspace.
    If missing, automatically clones the master workflow schema from the source workspace into theirs.
    """
    headers = {"Content-Type": "application/json"}
    effective_workspace = target_workspace.strip() if target_workspace and target_workspace.strip() else source_workspace
    
    # If the user provided their own workspace, check if workflow exists or fork it
    if effective_workspace != source_workspace:
        # 1. Check if user already has the workflow in their workspace
        check_url = f"https://api.roboflow.com/{effective_workspace}/workflows/{workflow_id}?api_key={api_key}"
        res = requests.get(check_url, headers=headers)
        if res.status_code == 200:
            return effective_workspace, f"Found existing workflow `{workflow_id}` in workspace `{effective_workspace}`."

        # 2. Fetch definition from master source workspace
        source_url = f"https://api.roboflow.com/{source_workspace}/workflows/{workflow_id}?api_key={api_key}"
        source_res = requests.get(source_url, headers=headers)
        
        if source_res.status_code == 200:
            # 3. Fork workflow into target workspace
            workflow_data = source_res.json()
            create_url = f"https://api.roboflow.com/{effective_workspace}/workflows?api_key={api_key}"
            create_res = requests.post(create_url, json=workflow_data, headers=headers)
            
            if create_res.status_code in [200, 201]:
                return effective_workspace, f"Successfully cloned `{workflow_id}` into workspace `{effective_workspace}`!"

    # Fallback to source workspace
    return source_workspace, f"Running inference against template workspace `{source_workspace}`."


def extract_images_from_uploads(uploaded_files):
    """
    Extracts raw image bytes while preserving relative folder hierarchy without altering image quality.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    extracted_images = []

    for file_item in uploaded_files:
        filename = file_item.name
        ext = os.path.splitext(filename)[1].lower()

        # Handle dropped ZIP archives containing folders/subfolders
        if ext == ".zip":
            try:
                with zipfile.ZipFile(file_item) as z:
                    for zip_info in z.infolist():
                        if not zip_info.is_dir():
                            zip_ext = os.path.splitext(zip_info.filename)[1].lower()
                            if zip_ext in valid_extensions:
                                img_bytes = z.read(zip_info.filename)
                                # Preserve full relative path
                                extracted_images.append((zip_info.filename, img_bytes))
            except Exception as e:
                st.error(f"Error reading ZIP file '{filename}': {e}")
        elif ext in valid_extensions:
            extracted_images.append((filename, file_item.getvalue()))

    return extracted_images


def run_one_image(filename, image_bytes, api_key, workspace, workflow_id):
    """
    Sends original full-detail image bytes to the Roboflow Workflow API.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    url = f"https://outline.roboflow.com/{workspace}/{workflow_id}?api_key={api_key}"
    payload = {
        "inputs": {
            "image": {"type": "base64", "value": b64_image}
        }
    }
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        row = {
            "Filename": filename, 
            "Status": "Success",
            "Timestamp": timestamp
        }
        
        if isinstance(data, list) and len(data) > 0:
            outputs = data[0].get("outputs", {})
        elif isinstance(data, dict):
            outputs = data.get("outputs", {})
        else:
            outputs = {}

        # Unpack all workflow predictions into scalar columns
        for k, v in outputs.items():
            row[k] = clean_scalar(v)

        return row
    except Exception as e:
        return {
            "Filename": filename,
            "Status": "Failed",
            "Timestamp": timestamp,
            "Error": str(e)
        }


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================

if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None
if "csv_results" not in st.session_state:
    st.session_state.csv_results = None
if "csv_summary" not in st.session_state:
    st.session_state.csv_summary = None
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None


# ==========================================
# 3. STREAMLIT UI & CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Boat Registration & Classification System",
    page_icon="🛥️",
    layout="wide"
)

st.title("🛥️ Boat Registration & Classification System")
st.markdown("Upload individual images, dropped folders, or ZIP archives for automated high-detail analysis.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Roboflow API Key", type="password")
    
    # User inputs THEIR OWN workspace ID (does not default to your workspace string)
    target_workspace = st.text_input(
        "Your Roboflow Workspace ID", 
        placeholder="e.g., my-personal-workspace",
        help="Enter your personal or team Roboflow workspace slug where the workflow will be hosted."
    )
    
    workflow_id = st.text_input("Workflow ID", value=DEFAULT_WORKFLOW)
    
    st.divider()
    max_workers = st.slider("Parallel Processing Threads", min_value=1, max_value=10, value=4)

# File / Folder Dropzone
st.subheader("📁 Input Files")
uploaded_items = st.file_uploader(
    "Drag and drop image files, folders, or ZIP archives here:",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "zip"],
    accept_multiple_files=True
)

# Optional Server Save Path
server_output_dir = st.text_input("Optional Server Save Path (Leave blank to download via browser):")


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

if st.button("🚀 Process Batch", type="primary", disabled=not (uploaded_items and api_key)):
    st.session_state.run_complete = False
    
    with st.status("Initializing workflow & preparing full-fidelity images...", expanded=True) as status:
        
        # Step A: Check / Fork Workflow into user's workspace
        st.write("Verifying Roboflow workflow access...")
        active_workspace, wf_msg = ensure_workflow_exists(
            api_key=api_key, 
            target_workspace=target_workspace, 
            source_workspace=SOURCE_WORKSPACE, 
            workflow_id=workflow_id
        )
        st.write(wf_msg)
        
        # Step B: Extract files
        image_batch = extract_images_from_uploads(uploaded_items)
        st.write(f"Total images identified for processing: **{len(image_batch)}**")

        if not image_batch:
            st.error("No valid image files found in the upload.")
            st.stop()

        status.update(label="Processing images through Roboflow API...", state="running")
        progress_bar = st.progress(0)
        
        results = []
        completed_count = 0
        total_count = len(image_batch)

        # Threaded Execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(run_one_image, fn, bytes_data, api_key, active_workspace, workflow_id): fn
                for fn, bytes_data in image_batch
            }

            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
                completed_count += 1
                progress_bar.progress(completed_count / total_count)

        status.update(label="Processing complete! Generating output packages...", state="complete")

    # Data Processing
    df_res = pd.DataFrame(results)
    
    # Summary Table Aggregation
    summary_data = {
        "Processing Date": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "Total Images Processed": [len(df_res)],
        "Successful Inferences": [len(df_res[df_res["Status"] == "Success"])],
        "Failed Inferences": [len(df_res[df_res["Status"] == "Failed"])]
    }
    df_sum = pd.DataFrame(summary_data)

    # 1. Generate CSV Bytes
    csv_res_bytes = df_res.to_csv(index=False).encode('utf-8')
    csv_sum_bytes = df_sum.to_csv(index=False).encode('utf-8')

    # 2. Generate Excel Workbook Bytes
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_res.to_excel(writer, sheet_name='Detailed Results', index=False)
        df_sum.to_excel(writer, sheet_name='Summary', index=False)
    excel_data = excel_buffer.getvalue()

    # 3. Generate ZIP Bundle Bytes
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("classification_results.csv", csv_res_bytes)
        zip_file.writestr("summary.csv", csv_sum_bytes)
        zip_file.writestr("full_report.xlsx", excel_data)
    zip_data = zip_buffer.getvalue()

    # Save to Server Path if defined
    if server_output_dir:
        os.makedirs(server_output_dir, exist_ok=True)
        df_res.to_csv(os.path.join(server_output_dir, "classification_results.csv"), index=False)
        df_sum.to_csv(os.path.join(server_output_dir, "summary.csv"), index=False)
        with open(os.path.join(server_output_dir, "full_report.xlsx"), "wb") as f:
            f.write(excel_data)
        st.success(f"Results persisted on server at: `{server_output_dir}`")

    # Update session state to persist data across downloads/re-runs
    st.session_state.df_results = df_res
    st.session_state.df_summary = df_sum
    st.session_state.csv_results = csv_res_bytes
    st.session_state.csv_summary = csv_sum_bytes
    st.session_state.excel_bytes = excel_data
    st.session_state.zip_bytes = zip_data
    st.session_state.run_complete = True


# ==========================================
# 5. PERSISTENT DISPLAY & DOWNLOAD SECTION
# ==========================================

if st.session_state.run_complete:
    st.divider()
    st.header("📊 Results & Downloads")

    tab1, tab2 = st.tabs(["Detailed Results", "Batch Summary"])

    with tab1:
        st.dataframe(st.session_state.df_results, use_container_width=True)

    with tab2:
        st.dataframe(st.session_state.df_summary, use_container_width=True)

    st.subheader("📥 Export Options")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="📄 Results CSV",
            data=st.session_state.csv_results,
            file_name="boat_classification_results.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="📊 Summary CSV",
            data=st.session_state.csv_summary,
            file_name="boat_classification_summary.csv",
            mime="text/csv"
        )

    with col3:
        st.download_button(
            label="📘 Excel Report",
            data=st.session_state.excel_bytes,
            file_name="boat_classification_full_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col4:
        st.download_button(
            label="📦 Complete ZIP Package",
            data=st.session_state.zip_bytes,
            file_name="boat_classification_export.zip",
            mime="application/zip"
        )
