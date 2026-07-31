import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
import zipfile
import pandas as pd
import requests
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Boat Registration & Classification",
    page_icon="🛥️",
    layout="wide"
)

# ==========================================
# MASTER WORKFLOW CONFIGURATION
# ==========================================
# Roboflow /forkWorkflow requires URL Slugs, not internal IDs
SOURCE_WORKSPACE_ID = "blakes-workspace-efak8"
SOURCE_WORKFLOW_ID = "boat-registration-and-classification-combination"

# Target workflow endpoint slug inside the user's workspace
TARGET_WORKFLOW_ID = "boat-registration-and-classification-combination"


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def auto_detect_workspace(user_api_key: str) -> str:
    """Detects the workspace slug associated with the provided API key."""
    url = f"https://api.roboflow.com/?api_key={user_api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("workspace", "")
    except Exception:
        pass
    return ""


def fork_workflow_to_workspace(user_api_key: str, user_workspace: str) -> bool:
    """Forks the master workflow template into the user's workspace using URL slugs."""
    url = f"https://api.roboflow.com/{user_workspace}/forkWorkflow"
    params = {"api_key": user_api_key}
    payload = {
        "source_workspace": SOURCE_WORKSPACE_ID,
        "source_workflow": SOURCE_WORKFLOW_ID,
        "name": "Boat Registration and Classification Combination",
        "url": TARGET_WORKFLOW_ID
    }
    
    try:
        response = requests.post(url, params=params, json=payload, timeout=15)
        
        # 200/201 indicates successful fork
        if response.status_code in [200, 201]:
            return True
            
        # 400/409 indicates it already exists in target workspace, which is fine to proceed
        if response.status_code in [400, 409]:
            return True
            
        st.error(f"Failed to setup workflow in `{user_workspace}`. Error: {response.text}")
        return False
    except Exception as e:
        st.error(f"Network error during setup: {str(e)}")
        return False


def run_workflow_inference(user_api_key: str, user_workspace: str, image_bytes: bytes) -> dict:
    """Executes the workflow using standard REST API calls (no external SDK required)."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    url = f"https://serverless.roboflow.com/{user_workspace}/{TARGET_WORKFLOW_ID}"
    params = {"api_key": user_api_key}
    payload = {
        "inputs": {
            "image": {"type": "base64", "value": base64_image}
        }
    }
    
    response = requests.post(url, params=params, json=payload, timeout=30)
    
    # Fallback endpoint if serverless route returns an error
    if response.status_code != 200:
        fallback_url = f"https://outline.roboflow.com/{user_workspace}/{TARGET_WORKFLOW_ID}"
        response = requests.post(fallback_url, params=params, json=payload, timeout=30)
        
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")


def extract_images_from_uploads(uploaded_files):
    """Extracts individual image files and unpacks ZIP archives."""
    extracted = []
    for file in uploaded_files:
        if file.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file) as z:
                    for filename in z.namelist():
                        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")) and not filename.startswith("__MACOSX"):
                            img_bytes = z.read(filename)
                            clean_name = os.path.basename(filename)
                            if clean_name:
                                extracted.append({"name": clean_name, "bytes": img_bytes})
            except Exception as e:
                st.error(f"Error extracting ZIP archive `{file.name}`: {str(e)}")
        else:
            extracted.append({"name": file.name, "bytes": file.getvalue()})
    return extracted


def flatten_json_outputs(data, prefix=""):
    """Flattens complex nested workflow output JSON into tabular rows."""
    flattened = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key_name = f"{prefix}_{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                flattened.update(flatten_json_outputs(v, key_name))
            else:
                flattened[key_name] = v
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            summaries = []
            for item in data:
                if "class" in item and "confidence" in item:
                    summaries.append(f"{item['class']} ({item['confidence']:.2f})")
                elif "text" in item:
                    summaries.append(str(item["text"]))
                else:
                    summaries.append(str(item))
            flattened[prefix] = " | ".join(summaries)
        else:
            flattened[prefix] = ", ".join([str(x) for x in data])
    else:
        flattened[prefix] = data
    return flattened


# ==========================================
# USER INTERFACE
# ==========================================
st.title("🛥️ Boat Registration & Classification Pipeline")
st.markdown("Batch process images through detection, registration extraction, and classification workflows.")

# Sidebar Credentials & Settings
st.sidebar.header("🔑 Credentials")
api_key = st.sidebar.text_input("Roboflow API Key", type="password", help="Enter your Roboflow Private API Key")

workspace_id = ""
if api_key:
    detected_workspace = auto_detect_workspace(api_key)
    workspace_id = st.sidebar.text_input("Workspace Slug", value=detected_workspace, help="Your Roboflow Workspace Slug")
else:
    workspace_id = st.sidebar.text_input("Workspace Slug")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Performance")
max_workers = st.sidebar.slider("Parallel Threads", min_value=1, max_value=10, value=3)

# Main File Input
uploaded_files = st.file_uploader(
    "Upload boat images or ZIP archives containing images...", 
    type=["jpg", "jpeg", "png", "webp", "zip"], 
    accept_multiple_files=True
)

if st.button("Process Batch", type="primary"):
    if not api_key or not workspace_id:
        st.warning("Please provide your Roboflow API Key and Workspace Slug in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload image files or a ZIP archive to process.")
    else:
        with st.spinner("Setting up workflow endpoint in your workspace..."):
            setup_ok = fork_workflow_to_workspace(api_key, workspace_id.strip())
            
        if setup_ok:
            images = extract_images_from_uploads(uploaded_files)
            if not images:
                st.error("No valid image files found in upload.")
            else:
                st.info(f"Loaded {len(images)} images for processing.")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                results_list = []
                completed = 0
                
                # Concurrent Batch Processing
                def process_task(item):
                    try:
                        res = run_workflow_inference(api_key, workspace_id.strip(), item["bytes"])
                        return {"name": item["name"], "bytes": item["bytes"], "data": res, "error": None}
                    except Exception as err:
                        return {"name": item["name"], "bytes": item["bytes"], "data": None, "error": str(err)}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_task, img) for img in images]
                    for future in as_completed(futures):
                        res = future.result()
                        results_list.append(res)
                        completed += 1
                        progress_bar.progress(completed / len(images))
                        status_text.text(f"Processed {completed} of {len(images)} images...")

                st.success("Batch processing complete!")
                
                # Store in session state for tab inspection
                st.session_state["results"] = results_list

# Display Results
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    
    tab1, tab2, tab3 = st.tabs(["📊 Table View", "🖼️ Detailed Inspection", "📥 Export Data"])
    
    # Flatten Data for Table View
    table_rows = []
    for item in results:
        row = {"Filename": item["name"]}
        if item["error"]:
            row["Status"] = "Failed"
            row["Error"] = item["error"]
        else:
            row["Status"] = "Success"
            flat_outputs = flatten_json_outputs(item["data"])
            row.update(flat_outputs)
        table_rows.append(row)
        
    df = pd.DataFrame(table_rows)
    
    with tab1:
        st.dataframe(df, use_container_width=True)
        
    with tab2:
        for item in results:
            with st.expander(f"📷 {item['name']} - {item.get('Status', 'Success')}"):
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(item["bytes"], caption=item["name"], use_column_width=True)
                with c2:
                    if item["error"]:
                        st.error(item["error"])
                    else:
                        st.json(item["data"])
                        
    with tab3:
        st.markdown("### Download Results")
        col_dl1, col_dl2 = st.columns(2)
        
        # CSV Export
        csv_data = df.to_csv(index=False).encode('utf-8')
        col_dl1.download_button("Download as CSV", data=csv_data, file_name="boat_pipeline_results.csv", mime="text/csv")
        
        # Excel Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Boat Pipeline")
        col_dl2.download_button(
            "Download as Excel", 
            data=excel_buffer.getvalue(), 
            file_name="boat_pipeline_results.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
