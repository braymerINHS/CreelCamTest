import io
import csv
import base64
import time
import json
import zipfile
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
import streamlit as st
from roboflow.adapters import rfapi


# =========================
# SOURCE WORKFLOW TEMPLATE
# =========================
# This is YOUR public/shared source workflow.
# The app will fork/copy this into the user's workspace, then run THEIR copy.
SOURCE_WORKSPACE_NAME = "blakes-workspace-efak8"
SOURCE_WORKFLOW_ID = "boat-registration-and-classification-combination"

# This is the workflow name/slug that will be created inside each user's workspace.
TARGET_WORKFLOW_NAME = "Boat Registration and Classification Tool"
TARGET_WORKFLOW_ID = "boat-registration-and-classification-tool"

API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_BASE = "https://api.roboflow.com"

USE_CACHE = False

RESULTS_FOLDER_NAME = "RoboFlow_Results"


# =========================
# OUTPUT COLUMNS
# Must match local batch script
# =========================
RESULTS_FIELDNAMES = [
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

SUMMARY_FIELDNAMES = [
    "run_id",
    "run_started_at",
    "date",
    "registration_number",
    "image_count",
    "images",
]


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Boat Registration & Classification Tool",
    page_icon="🚤",
    layout="wide"
)

st.title("🚤 Boat Registration & Classification Tool")
st.write(
    "Upload boat images below. The app will install the workflow into your Roboflow workspace "
    "if needed, then run the analysis using your Roboflow credits."
)

st.sidebar.header("🔑 Roboflow Credentials")

user_api_key = st.sidebar.text_input(
    "Roboflow Private API Key",
    type="password",
    help=(
        "Enter the private API key for the workspace you want billed. "
        "The key needs workflow:read, workflow:create, workflow:update, and model:infer permissions."
    )
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Processing Settings")

workers = st.sidebar.slider("Concurrent Workers", min_value=1, max_value=5, value=3)
max_retries = st.sidebar.number_input("Max Retries per Image", value=3, min_value=1)
retry_sleep_seconds = st.sidebar.number_input("Retry Delay (seconds)", value=8, min_value=1)

st.sidebar.markdown("---")
st.sidebar.header("🧩 Workflow Setup")
st.sidebar.caption(
    "The app will look for a workflow named "
    f"`{TARGET_WORKFLOW_ID}` in the user's workspace. If missing, it will fork your shared source workflow."
)


# =========================
# LOCAL RESULTS FOLDER
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / RESULTS_FOLDER_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# HELPER FUNCTIONS
# =========================
def clean_scalar(value):
    """
    Convert workflow outputs to spreadsheet-safe scalar values.
    Avoids huge JSON/base64 blobs in clean CSV / Excel output.
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
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        return result
    return {}


def extract_workflow_url_from_response(value):
    """
    Roboflow workflow API / SDK responses can vary slightly.
    This helper tries to find the workflow URL slug robustly.
    """
    if not isinstance(value, dict):
        return ""

    for key in ["url", "workflowUrl", "workflow_url"]:
        if value.get(key):
            return value[key]

    workflows = value.get("workflows")
    if isinstance(workflows, dict):
        for key in ["url", "workflowUrl", "workflow_url"]:
            if workflows.get(key):
                return workflows[key]

    if isinstance(workflows, list) and workflows:
        first = workflows[0]
        if isinstance(first, dict):
            for key in ["url", "workflowUrl", "workflow_url"]:
                if first.get(key):
                    return first[key]

    return ""


def get_workspace_from_api_key(api_key):
    """
    The root Roboflow API endpoint identifies the workspace for a valid API key.
    """
    response = requests.get(
        f"{ROBOFLOW_API_BASE}/",
        params={"api_key": api_key},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Could not validate Roboflow API key: HTTP {response.status_code}: {response.text}")

    data = response.json()
    workspace = data.get("workspace")

    if not workspace:
        raise RuntimeError(
            "Could not determine workspace from this API key. "
            "Please check that the key is valid and belongs to a Roboflow workspace."
        )

    return workspace


def list_user_workflows(api_key, workspace_name):
    response = requests.get(
        f"{ROBOFLOW_API_BASE}/{workspace_name}/workflows",
        params={"api_key": api_key},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Could not list workflows: HTTP {response.status_code}: {response.text}")

    data = response.json()
    return data.get("workflows", [])


def find_existing_workflow(api_key, workspace_name, workflow_url):
    workflows = list_user_workflows(api_key, workspace_name)

    for workflow in workflows:
        if workflow.get("url") == workflow_url:
            return workflow

    return None


def ensure_workflow_installed(api_key):
    """
    Finds or forks the workflow into the user's workspace.
    Returns (workspace_name, workflow_id_to_run, install_status_message).
    """
    workspace_name = get_workspace_from_api_key(api_key)

    existing = find_existing_workflow(api_key, workspace_name, TARGET_WORKFLOW_ID)
    if existing:
        return (
            workspace_name,
            existing.get("url", TARGET_WORKFLOW_ID),
            f"Found existing workflow copy in `{workspace_name}`: `{existing.get('url', TARGET_WORKFLOW_ID)}`"
        )

    forked = rfapi.fork_workflow(
        api_key=api_key,
        workspace_url=workspace_name,
        source_workspace=SOURCE_WORKSPACE_NAME,
        source_workflow=SOURCE_WORKFLOW_ID,
        name=TARGET_WORKFLOW_NAME,
        url=TARGET_WORKFLOW_ID,
    )

    workflow_url = extract_workflow_url_from_response(forked)

    if not workflow_url:
        # Fallback: list again after fork.
        existing_after_fork = find_existing_workflow(api_key, workspace_name, TARGET_WORKFLOW_ID)
        if existing_after_fork:
            workflow_url = existing_after_fork.get("url", TARGET_WORKFLOW_ID)

    if not workflow_url:
        raise RuntimeError(
            "The workflow fork call completed, but I could not determine the new workflow URL. "
            f"Raw response: {forked}"
        )

    return (
        workspace_name,
        workflow_url,
        f"Forked workflow into `{workspace_name}` as `{workflow_url}`"
    )


def extract_date_only(date_and_time):
    text = str(date_and_time or "").strip()
    if not text:
        return ""
    return text.split()[0] if text.split() else ""


def run_one_image(
    uploaded_file,
    api_key,
    workspace_name,
    workflow_id,
    run_id,
    run_started_at,
    retries,
    sleep_sec,
):
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            url = f"{API_URL}/infer/workflows/{workspace_name}/{workflow_id}"
            payload = {
                "api_key": api_key,
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": base64_image,
                    }
                },
                "use_cache": USE_CACHE,
            }

            response = requests.post(url, json=payload, timeout=120)

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            data = response.json()
            outputs = data.get("outputs", data)
            output = first_result(outputs)

            return {
                "run_id": run_id,
                "run_started_at": run_started_at.isoformat(timespec="seconds"),
                "source_image": uploaded_file.name,
                "file_name": uploaded_file.name,

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
                "attempts": attempt,
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(sleep_sec)

    return {
        "run_id": run_id,
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "source_image": uploaded_file.name,
        "file_name": uploaded_file.name,
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
        "workflow_error": last_error,
        "attempts": retries,
    }


def generate_daily_summary_rows(rows, run_id, run_started_at):
    summary = defaultdict(lambda: {
        "run_id": run_id,
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
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

    return summary_rows


def dataframe_to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(
        buffer,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def build_excel_bytes(df_results, df_summary):
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Clean Results", index=False)
        df_summary.to_excel(writer, sheet_name="Daily Summary", index=False)

    return excel_buffer.getvalue()


def build_zip_bytes(files):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name, file_bytes in files.items():
            zf.writestr(file_name, file_bytes)

    return zip_buffer.getvalue()


def save_outputs_to_results_folder(run_id, results_csv_bytes, summary_csv_bytes, excel_bytes):
    results_csv_path = RESULTS_DIR / f"roboflow_results_clean_{run_id}.csv"
    summary_csv_path = RESULTS_DIR / f"registration_daily_summary_{run_id}.csv"
    excel_path = RESULTS_DIR / f"boat_registration_report_{run_id}.xlsx"

    results_csv_path.write_bytes(results_csv_bytes)
    summary_csv_path.write_bytes(summary_csv_bytes)
    excel_path.write_bytes(excel_bytes)

    return results_csv_path, summary_csv_path, excel_path


# =========================
# MAIN APP
# =========================
uploaded_files = st.file_uploader(
    "Drag & drop boat images here",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"Staged **{len(uploaded_files)}** images for processing.")

    if st.button("Start Batch Analysis", type="primary"):
        if not user_api_key:
            st.error("Please enter your Roboflow Private API Key in the sidebar to proceed.")
            st.stop()

        with st.spinner("Checking your Roboflow workspace and installing the workflow if needed..."):
            try:
                user_workspace_name, user_workflow_id, setup_message = ensure_workflow_installed(user_api_key)
                st.success(setup_message)
            except Exception as e:
                st.error("Could not install or find the workflow in your Roboflow workspace.")
                st.code(f"{type(e).__name__}: {e}")
                st.info(
                    "Most common causes: your API key does not have workflow permissions, "
                    "or the source workflow is not publicly forkable/shared."
                )
                st.stop()

        run_started_at = datetime.now()
        run_id = run_started_at.strftime("%Y-%m-%d_%H-%M-%S")

        st.subheader("Progress Monitor")
        st.caption(f"Running workflow `{user_workspace_name}/{user_workflow_id}` using the provided API key.")

        progress_bar = st.progress(0)
        status_text = st.empty()

        rows = []
        completed = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(
                    run_one_image,
                    file,
                    user_api_key,
                    user_workspace_name,
                    user_workflow_id,
                    run_id,
                    run_started_at,
                    max_retries,
                    retry_sleep_seconds,
                ): file
                for file in uploaded_files
            }

            for future in as_completed(future_to_file):
                file = future_to_file[future]
                completed += 1

                try:
                    row = future.result()
                except Exception:
                    row = {
                        "run_id": run_id,
                        "run_started_at": run_started_at.isoformat(timespec="seconds"),
                        "source_image": file.name,
                        "file_name": file.name,
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
                        "workflow_error": traceback.format_exc(),
                        "attempts": max_retries,
                    }

                rows.append(row)

                elapsed = time.time() - start_time
                avg = elapsed / completed
                eta_seconds = int(avg * (len(uploaded_files) - completed))

                progress_bar.progress(completed / len(uploaded_files))
                status_text.text(
                    f"[{completed}/{len(uploaded_files)}] "
                    f"Processed {file.name} | "
                    f"Status: {row.get('status')} | "
                    f"Registration: {row.get('registration_number')} | "
                    f"ETA: {eta_seconds}s"
                )

        status_text.success("All images processed.")

        rows = sorted(rows, key=lambda r: r.get("source_image", ""))

        summary_rows = generate_daily_summary_rows(rows, run_id, run_started_at)

        df_results = pd.DataFrame(rows).reindex(columns=RESULTS_FIELDNAMES)
        df_summary = pd.DataFrame(summary_rows).reindex(columns=SUMMARY_FIELDNAMES)

        results_csv_bytes = dataframe_to_csv_bytes(df_results)
        summary_csv_bytes = dataframe_to_csv_bytes(df_summary)
        excel_bytes = build_excel_bytes(df_results, df_summary)

        results_csv_name = f"roboflow_results_clean_{run_id}.csv"
        summary_csv_name = f"registration_daily_summary_{run_id}.csv"
        excel_name = f"boat_registration_report_{run_id}.xlsx"
        zip_name = f"boat_registration_outputs_{run_id}.zip"

        results_csv_path, summary_csv_path, excel_path = save_outputs_to_results_folder(
            run_id,
            results_csv_bytes,
            summary_csv_bytes,
            excel_bytes,
        )

        tab1, tab2 = st.tabs(["Clean Results Sheet", "Daily Registration Summary Sheet"])

        with tab1:
            st.dataframe(df_results, use_container_width=True)

        with tab2:
            st.dataframe(df_summary, use_container_width=True)

        st.subheader("Downloads")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="📥 Download Clean Results CSV",
                data=results_csv_bytes,
                file_name=results_csv_name,
                mime="text/csv",
                type="primary",
            )

        with col2:
            st.download_button(
                label="📥 Download Daily Summary CSV",
                data=summary_csv_bytes,
                file_name=summary_csv_name,
                mime="text/csv",
            )

        with col3:
            st.download_button(
                label="📥 Download Excel Workbook",
                data=excel_bytes,
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        zip_bytes = build_zip_bytes({
            results_csv_name: results_csv_bytes,
            summary_csv_name: summary_csv_bytes,
            excel_name: excel_bytes,
        })

        st.download_button(
            label="📦 Download All Outputs as ZIP",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
        )

        st.success(
            "Output files were also written on the app server under "
            f"`{RESULTS_DIR}`. If this is hosted Streamlit, users should use the download buttons."
        )
