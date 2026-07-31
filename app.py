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
# MASTER WORKFLOW CONFIGURATION
# ==========================================
# Roboflow /forkWorkflow requires exact URL Slugs from your browser address bar
SOURCE_WORKSPACE_ID = "blakes-workspace-efak8"
SOURCE_WORKFLOW_ID = "boat-registration-and-classification-combination"

# Target named endpoint inside the user's workspace
TARGET_WORKFLOW_ID = "boat-registration-and-classification-combination"


# ==========================================
# 1. HELPER FUNCTIONS & AUTO-FORK ENGINE
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


def get_user_workspace(api_key):
    """
    Automatically detects the user's primary Roboflow Workspace Slug using their API key.
    """
    try:
        res = requests.get(f"https://api.roboflow.com/?api_key={api_key}", timeout=10)
        if res.status_code == 200:
            data = res.json()
            if "workspace" in data and isinstance(data["workspace"], str):
                return data["workspace"]
            elif "workspaces" in data:
                workspaces = data["workspaces"]
                if isinstance(workspaces, dict) and len(workspaces) > 0:
                    return list(workspaces.keys())[0]
                elif isinstance(workspaces, list) and len(workspaces) > 0:
                    ws = workspaces[0]
                    return ws.get("url") or ws.get("slug") or ws.get("id")
    except Exception:
        pass
    return None


def ensure_user_has_forked_workflow(user_workspace, api_key):
    """
    1. Checks if the workflow exists in the user's workspace.
    2. If missing, auto-forks the master workflow template using Roboflow's fork API.
    """
    headers = {"Content-Type": "application/json"}
    
    # Step A: Check if workflow already exists in user's workspace
    check_url = f"https://api.roboflow.com/{user_workspace}/workflows/{TARGET_WORKFLOW_ID}?api_key={api_key}"
    res = requests.get(check_url, headers=headers)
    
    if res.status_code == 200:
        return True, f"Found active workflow in workspace `{user_workspace}`."

    # Step B: If missing, trigger programmatic fork using URL Slugs
    fork_url = f"https://api.roboflow.com/{user_workspace}/forkWorkflow?api_key={api_key}"
    payload = {
        "source_workspace": SOURCE_WORKSPACE_ID,
        "source_workflow": SOURCE_WORKFLOW_ID,
        "name": TARGET_WORKFLOW_ID,
        "url": TARGET_WORKFLOW_ID
    }
    
    fork_res = requests.post(fork_url, json=payload, headers=headers)
    
    # 200/201 or 400/409 (already exists) indicate the workflow is ready
    if fork_res.status_code in [200, 201] or "already exists" in fork_res.text.lower() or fork_res.status_code in [400, 409]:
        return True, f"Successfully set up workflow in workspace `{user_workspace}`!"
    else:
        return False, f"❌ Failed to fork workflow into workspace `{user_workspace}`. Error: {fork_res.text}"


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


def run_one_image(filename, image_bytes, api_key, workspace):
    """
    Sends original full-detail image bytes to the user's workspace workflow API.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    url = f"https://outline.roboflow.com/{workspace}/{TARGET_WORKFLOW_ID}?api_key={api_key}"
    payload = {
        "inputs": {
            "image": {"type": "base64", "value": b64_image}
        }
    }
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        response = requests.post(url, json=payload, timeout=120)
        
        # Fallback endpoint check if primary returns 404
        if response.status_code != 200:
            alt_url = f"https://serverless.roboflow.com/{workspace}/{TARGET_WORKFLOW_ID}?api_key={api_key}"
            response = requests.post(alt_url, json=payload, timeout=120)

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

        # Dynamically unpack all output features returned by the workflow
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

# Minimalist Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Roboflow API Key", 
        type="password",
        help=(
            "How to get your API Key:\n"
            "1. Log into your Roboflow account at https://app.roboflow.com\n"
            "2. Click your workspace name or profile in the left/top sidebar.\n"
            "3. Select 'Workspace Settings' -> 'API Keys'.\n"
            "4. Copy your Private API Key and paste it here."
        )
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
    
    with st.status("Verifying workflow & workspace authorization...", expanded=True) as status:
        
        # Step A: Detect User Workspace
        user_workspace = get_user_workspace(api_key)
        if not user_workspace:
            st.error("❌ Could not verify Roboflow workspace. Please double check your API key.")
            st.stop()
            
        st.write(f"Detected Roboflow Workspace: `{user_workspace}`")
        
        # Step B: Auto-fork workflow into user's workspace if missing
        st.write("Verifying workflow setup...")
        fork_success, fork_msg = ensure_user_has_forked_workflow(user_workspace, api_key)
        
        if not fork_success:
            st.error(fork_msg)
            st.stop()
            
        st.write(fork_msg)
        
        # Step C: Extract files
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
                executor.submit(run_one_image, fn, bytes_data, api_key, user_workspace): fn
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
    
    summary_data = {
        "Processing Date": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "Total Images Processed": [len(df_res)],
        "Successful Inferences": [len(df_res[df_res["Status"] == "Success"])],
        "Failed Inferences": [len(df_res[df_res["Status"] == "Failed"])]
    }
    df_sum = pd.DataFrame(summary_data)

    csv_res_bytes = df_res.to_csv(index=False).encode('utf-8')
    csv_sum_bytes = df_sum.to_csv(index=False).encode('utf-8')

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_res.to_excel(writer, sheet_name='Detailed Results', index=False)
        df_sum.to_excel(writer, sheet_name='Summary', index=False)
    excel_data = excel_buffer.getvalue()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("classification_results.csv", csv_res_bytes)
        zip_file.writestr("summary.csv", csv_sum_bytes)
        zip_file.writestr("full_report.xlsx", excel_data)
    zip_data = zip_buffer.getvalue()

    if server_output_dir:
        os.makedirs(server_output_dir, exist_ok=True)
        df_res.to_csv(os.path.join(server_output_dir, "classification_results.csv"), index=False)
        df_sum.to_csv(os.path.join(server_output_dir, "summary.csv"), index=False)
        with open(os.path.join(server_output_dir, "full_report.xlsx"), "wb") as f:
            f.write(excel_data)
        st.success(f"Results persisted on server at: `{server_output_dir}`")

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
