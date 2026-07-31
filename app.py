import os
import requests
import streamlit as st
from inference_sdk import InferenceHTTPClient

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
# Roboflow /forkWorkflow requires URL Slugs, not alphanumeric IDs
SOURCE_WORKSPACE_ID = "blakes-workspace-efak8"
SOURCE_WORKFLOW_ID = "boat-registration-and-classification-combination"

# Target named endpoint inside the user's workspace
TARGET_WORKFLOW_ID = "boat-registration-and-classification-combination"


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def fork_workflow_to_workspace(user_api_key: str, user_workspace: str) -> bool:
    """
    Forks the master workflow from the source workspace into the user's target workspace.
    """
    url = f"https://api.roboflow.com/{user_workspace}/forkWorkflow"
    params = {"api_key": user_api_key}
    payload = {
        "source_workspace": SOURCE_WORKSPACE_ID,
        "source_workflow": SOURCE_WORKFLOW_ID,
        "name": "Boat Registration and Classification Combination",
        "url": TARGET_WORKFLOW_ID
    }
    
    try:
        response = requests.post(url, params=params, json=payload)
        
        # 200/201 indicates successful fork
        if response.status_code in [200, 201]:
            return True
            
        # If it already exists, Roboflow often returns 400/409, which is fine to treat as ready
        if response.status_code in [400, 409]:
            st.info("Workflow already exists in your workspace. Proceeding to execution.")
            return True
            
        st.error(f"Failed to fork workflow into workspace `{user_workspace}`. Error: {response.text}")
        return False
    except Exception as e:
        st.error(f"Network or request error during workflow fork: {str(e)}")
        return False


def run_workflow_inference(user_api_key: str, user_workspace: str, image_file) -> dict:
    """
    Executes the workflow against the uploaded image using the Inference HTTP Client.
    """
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=user_api_key
    )
    
    # Save uploaded file temporarily to pass path to the inference client
    temp_filename = f"temp_{image_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(image_file.getbuffer())
        
    try:
        # Construct target workflow ID path: workspace/workflow_slug
        workflow_path = f"{user_workspace}/{TARGET_WORKFLOW_ID}"
        result = client.run_workflow(
            workspace_name=user_workspace,
            workflow_id=TARGET_WORKFLOW_ID,
            images={"image": temp_filename}
        )
        return result
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ==========================================
# USER INTERFACE
# ==========================================
st.title("🛥️ Boat Registration & Classification Pipeline")
st.markdown("Upload images to process through the automated detection and classification workflow.")

# Sidebar - Credentials
st.sidebar.header("Roboflow Credentials")
api_key = st.sidebar.text_input("Roboflow API Key", type="password", help="Enter your Roboflow Private API Key")
workspace_id = st.sidebar.text_input("Workspace Slug", help="Your workspace URL slug (e.g. blakes-workspace-ng75r)")

st.sidebar.markdown("---")
st.sidebar.caption("Master Template Workspace: `" + SOURCE_WORKSPACE_ID + "`")

# Main Content - File Upload
uploaded_files = st.file_uploader("Choose boat images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Process Batch", type="primary"):
    if not api_key or not workspace_id:
        st.warning("Please provide both your API Key and Workspace Slug in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one image to process.")
    else:
        with st.spinner("Ensuring workflow template exists in your workspace..."):
            fork_success = fork_workflow_to_workspace(api_key, workspace_id.strip())
            
        if fork_success:
            st.success("Workspace ready! Running inference...")
            
            for file in uploaded_files:
                st.subheader(f"Results for: {file.name}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(file, caption="Uploaded Image", use_column_width=True)
                    
                with col2:
                    with st.spinner("Processing image..."):
                        try:
                            output = run_workflow_inference(api_key, workspace_id.strip(), file)
                            st.json(output)
                        except Exception as e:
                            st.error(f"Inference execution failed: {str(e)}")
