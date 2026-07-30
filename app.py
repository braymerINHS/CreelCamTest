import io
import base64
import time
import traceback
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
import streamlit as st

# =========================
# CONFIGURATION & CONSTANTS
# =========================
# Public Roboflow Workflow hosted under your workspace
WORKSPACE_NAME = "blakes-workspace-efak8"
WORKFLOW_ID = "boat-registration-and-classification-combination"

# =========================
# PAGE SETUP & SIDEBAR
# =========================
st.set_page_config(
    page_title="Boat Registration & Classification Tool",
    page_icon="🚤",
    layout="wide"
)

st.title("🚤 Boat Registration & Classification Tool")
st.write("Drag and drop your image dataset below to run automated registration OCR and boat classification.")

st.sidebar.header("🔑 Roboflow Credentials")

# Support both Streamlit Secrets (for optional hosted keys) and manual user input
secret_key = st.secrets.get("ROBOFLOW_API_KEY", "") if hasattr(st, "secrets") else ""

if secret_key:
    st.sidebar.success("✅ Connected via Hosted Workspace API Key")
    user_api_key = secret_key
else:
    user_api_key = st.sidebar.text_input(
        "Roboflow Private API Key",
        type="password",
        help="Enter your API key from app.roboflow.com/settings/api. Credits will be billed directly to your account."
    )

st.sidebar.markdown("---")
st.sidebar.header("⚡ Processing Settings")
workers = st.sidebar.slider("Concurrent Workers", min_value=1, max_value=5, value=3)
max_retries = st.sidebar.number_input("Max Retries per Image", value=3, min_value=1)
retry_sleep_seconds = st.sidebar.number_input("Retry Delay (seconds)", value=8, min_value=1)

# =========================
# HELPER FUNCTIONS
# =========================
def clean_scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ").strip()
    if isinstance(value, list):
        cleaned_items = [str(clean_scalar(item)) for item in value if clean_scalar(item) != ""]
        return " | ".join(cleaned_items)
    if isinstance(value, dict):
        if value.get("type") == "base64":
            return "[base64 image omitted]"
        if "predictions" in value:
            preds = value.get("predictions") or []
            return f"[{len(preds)} predictions omitted]"
        preferred_keys = [
            "boat_type", "confidence", "reason",
            "approved_identifiers_found", "unapproved_observations_report",
            "output", "parsed_output", "result", "text",
        ]
        parts = []
        for key in preferred_keys:
            if key in value:
                cleaned = clean_scalar(value.get(key))
                if cleaned != "":
                    parts.append(f"{key}: {cleaned}")
        return " | ".join(parts) if parts else "[dict omitted]"
    return str(value).replace("\r", " ").replace("\n", " ").strip()

def first_result(result):
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        return result
    return {}

def extract_date_only(date_and_time):
    text = str(date_and_time or "").strip()
    return text.split()[0] if text else ""

def run_one_image(uploaded_file, api_key, run_id, run_started_at, retries, sleep_sec):
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            # Read and encode image file to base64
            image_bytes = uploaded_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            url = f"https://serverless.roboflow.com/infer/workflows/{WORKSPACE_NAME}/{WORKFLOW_ID}"
            payload = {
                "api_key": api_key,
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": base64_image
                    }
                }
            }

            response = requests.post(url, json=payload, timeout=60)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            data = response.json()
            outputs = data.get("outputs", data)
            output = first_result(outputs)

            return {
                "run_id": run_id,
                "run_started_at": run_started_at.isoformat(timespec="seconds"),
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
        "file_name": uploaded_file.name,
        "status": "workflow_error",
        "workflow_error": last_error,
        "attempts": retries,
    }

def generate_daily_summary(rows, run_id, run_started_at):
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
    return pd.DataFrame(summary_rows)

# =========================
# MAIN APP UPLOADER & RUNNER
# =========================
uploaded_files = st.file_uploader(
    "Drag & drop boat images here",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Staged **{len(uploaded_files)}** images for processing.")

    if st.button("Start Batch Analysis", type="primary"):
        if not user_api_key:
            st.error("Please enter your Roboflow Private API Key in the sidebar to proceed.")
            st.stop()

        run_started_at = datetime.now()
        run_id = run_started_at.strftime("%Y-%m-%d_%H-%M-%S")

        st.subheader("Progress Monitor")
        progress_bar = st.progress(0)
        status_text = st.empty()

        rows = []
        completed = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(
                    run_one_image, file, user_api_key, 
                    run_id, run_started_at, max_retries, retry_sleep_seconds
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
                        "file_name": file.name,
                        "status": "script_error",
                        "workflow_error": traceback.format_exc(),
                    }
                rows.append(row)

                elapsed = time.time() - start_time
                avg = elapsed / completed
                eta_seconds = int(avg * (len(uploaded_files) - completed))

                progress_bar.progress(completed / len(uploaded_files))
                status_text.text(
                    f"[{completed}/{len(uploaded_files)}] Processed {file.name} | "
                    f"Status: {row.get('status')} | ETA: {eta_seconds}s"
                )

        status_text.success("All images processed successfully!")

        # Compile Excel Output
        rows = sorted(rows, key=lambda r: r.get("file_name", ""))
        df_results = pd.DataFrame(rows)
        df_summary = generate_daily_summary(rows, run_id, run_started_at)

        tab1, tab2 = st.tabs(["Clean Results Sheet", "Daily Registration Summary Sheet"])
        with tab1:
            st.dataframe(df_results, use_container_width=True)
        with tab2:
            st.dataframe(df_summary, use_container_width=True)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="Clean Results", index=False)
            df_summary.to_excel(writer, sheet_name="Daily Summary", index=False)

        st.download_button(
            label="📥 Download Complete Excel Workbook (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"boat_registration_report_{run_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
