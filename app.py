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
WORKFLOW_ID = "boat-registration-and-classification-combination"


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

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


def get_visitor_workspace_id(user_api_key):
    """
    Automatically retrieves the visitor's Roboflow workspace handle from their API Key.
    """
    url = f"https://api.roboflow.com/?api_key={user_api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    
    if "workspace" in data and data["workspace"]:
        return data["workspace"]
    elif "workspaces" in data:
        ws = data["workspaces"]
        if isinstance(ws, dict) and len(ws) > 0:
            return list(ws.keys())[0]
        elif isinstance(ws, list) and len(ws) > 0:
            return ws[0].get("url") or ws[0].get("id")
            
    raise Exception("Could not automatically determine workspace handle from this API Key.")


def auto_deploy_workflow_to_visitor(visitor_workspace, user_api_key):
    """
    Reads workflow.json from disk and deploys/syncs it to the visitor's workspace.
    Gracefully handles existing workflows (200, 201, 409 Conflict).
    """
    json_path = os.path.join(os.path.dirname(__file__), "workflow.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Could not find 'workflow.json' in the app directory. "
            "Please ensure your workflow.json file is placed in the same folder as app.py."
        )
        
    with open(json_path, "r", encoding="utf-8") as f:
        workflow_spec = json.load(f)

    deploy_url = f"https://api.roboflow.com/{visitor_workspace}/workflows/{WORKFLOW_ID}?api_key={user_api_key}"
    
    # Send POST request to deploy/sync workflow definition
    deploy_resp = requests.post(deploy_url, json=workflow_spec, timeout=20)
    
    # Status Codes: 200 (OK), 201 (Created), 409 (Already Exists) are all success states
    if deploy_resp.status_code not in [200, 201, 409]:
        # Fallback to PUT if POST returns an update requirement
        put_resp = requests.put(deploy_url, json=workflow_spec, timeout=20)
        if put_resp.status_code not in [200, 201, 409]:
            deploy_resp.raise_for_status()


def extract_images_from_uploads(uploaded_files):
    """Extracts raw image bytes while preserving folder structures."""
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


def run_one_image(filename, image_bytes, visitor_workspace, user_api_key):
    """
    Executes the workflow endpoint hosted inside the VISITOR'S workspace.
    Guarantees billing is charged to the visitor's account quota.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    url = f"https://outline.roboflow.com/{visitor_workspace}/{WORKFLOW_ID}?api_key={user_api_key}"
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
st.markdown("Upload individual images, dropped folders, or ZIP archives for automated analysis.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Roboflow API Key", 
        type="password",
        help="Enter your personal Roboflow Private API Key. Usage will be billed to your account."
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
    
    with st.status("Initializing automated workflow session...", expanded=True) as status:
        
        # Step A: Resolve visitor workspace ID from API key
        status.update(label="Authenticating key & identifying your workspace...", state="running")
        try:
            visitor_workspace = get_visitor_workspace_id(api_key.strip())
            st.write(f"Target Workspace Identified: **`{visitor_workspace}`**")
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.stop()

        # Step B: Deploy or sync workflow definition using local workflow.json
        status.update(label=f"Syncing workflow into workspace '{visitor_workspace}'...", state="running")
        try:
            auto_deploy_workflow_to_visitor(visitor_workspace, api_key.strip())
            st.write(f"Workflow verified in workspace **`{visitor_workspace}`**.")
        except Exception as e:
            st.warning(f"Note on workflow deployment: {e}. Attempting direct inference...")

        # Step C: Extract upload files
        image_batch = extract_images_from_uploads(uploaded_items)
        st.write(f"Total images identified for processing: **{len(image_batch)}**")

        if not image_batch:
            st.error("No valid image files found in the upload.")
            st.stop()

        status.update(label="Processing images (billed to your account)...", state="running")
        progress_bar = st.progress(0)
        
        results = []
        completed_count = 0
        total_count = len(image_batch)

        # Threaded Execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    run_one_image, 
                    fn, 
                    bytes_data, 
                    visitor_workspace, 
                    api_key.strip()
                ): fn for fn, bytes_data in image_batch
            }

            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
                completed_count += 1
                progress_bar.progress(completed_count / total_count)

        status.update(label="Processing complete! Generating export packages...", state="complete")

    # Data Formatting
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
