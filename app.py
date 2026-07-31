import base64
import datetime
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Keep this false while testing workflow edits.
USE_CACHE = False

# Keep retries modest because the workflow is heavy.
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 8

DETAIL_COLUMNS = [
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

SUMMARY_COLUMNS = [
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
    "image_names",
    "boat_types",
]


# ==========================================
# 2. HELPER FUNCTIONS
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

    # Supports either:
    # 1. Raw workflow spec
    # 2. Exported wrapper shaped like {"specification": {...}}
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
                    inp_name = inp.get("name", "")

                    if "image" in inp_type or inp_name in ["image", "images", "input_image"]:
                        return inp_name or "image"

    return "image"


def clean_scalar(value):
    """
    Converts workflow outputs to spreadsheet-safe scalar values.
    Avoids huge JSON, image blobs, and prediction blobs in clean outputs.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        value = value.replace("\r", " ").replace("\n", " ").strip()

        if len(value) > 1000 or value.startswith("data:image"):
            return "[IMAGE/LONG BLOB DATA]"

        return value

    if isinstance(value, list):
        cleaned_items = []

        for item in value:
            cleaned = clean_scalar(item)
            if cleaned != "":
                cleaned_items.append(str(cleaned))

        return " | ".join(cleaned_items)

    if isinstance(value, dict):
        # Do not write base64 images into clean spreadsheets.
        if value.get("type") == "base64":
            return "[base64 image omitted]"

        # Do not write full prediction blobs into clean spreadsheets.
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


def unwrap_workflow_output(data):
    """
    Normalizes Roboflow workflow responses into the first output dictionary.

    Handles common shapes:
    1. [{"registration_number": "..."}]
    2. {"outputs": [{"registration_number": "..."}]}
    3. {"outputs": {"registration_number": "..."}}
    4. {"registration_number": "..."}
    """
    if isinstance(data, list):
        if not data:
            return {}

        first = data[0]

        if isinstance(first, dict) and "outputs" in first:
            outputs = first.get("outputs")

            if isinstance(outputs, list) and outputs:
                return outputs[0] if isinstance(outputs[0], dict) else {}

            if isinstance(outputs, dict):
                return outputs

        return first if isinstance(first, dict) else {}

    if isinstance(data, dict):
        if "outputs" in data:
            outputs = data.get("outputs")

            if isinstance(outputs, list) and outputs:
                return outputs[0] if isinstance(outputs[0], dict) else {}

            if isinstance(outputs, dict):
                return outputs

        return data

    return {}


def extract_images_from_uploads(uploaded_files):
    """Extracts raw image bytes while preserving folder structures and unpacking ZIPs."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    extracted_images = []

    for file_item in uploaded_files:
        filename = file_item.name
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".zip":
            try:
                with zipfile.ZipFile(file_item) as z:
                    for zip_info in z.infolist():
                        if zip_info.is_dir():
                            continue

                        if zip_info.filename.startswith("__MACOSX"):
                            continue

                        zip_ext = os.path.splitext(zip_info.filename)[1].lower()

                        if zip_ext in valid_extensions:
                            img_bytes = z.read(zip_info.filename)
                            clean_name = os.path.basename(zip_info.filename)

                            if clean_name:
                                extracted_images.append(
                                    {
                                        "file_name": clean_name,
                                        "source_image": zip_info.filename,
                                        "image_bytes": img_bytes,
                                    }
                                )

            except Exception as e:
                st.error(f"Error reading ZIP file '{filename}': {e}")

        elif ext in valid_extensions:
            extracted_images.append(
                {
                    "file_name": filename,
                    "source_image": filename,
                    "image_bytes": file_item.getvalue(),
                }
            )

    return extracted_images


def build_error_row(
    run_id,
    run_started_at,
    source_image,
    file_name,
    status,
    error,
    attempts,
):
    """Creates a standardized failed row with the same columns as successful rows."""
    return {
        "run_id": run_id,
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "source_image": source_image,
        "file_name": file_name,
        "status": status,
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
        "workflow_error": error,
        "attempts": attempts,
    }


def run_one_image(
    file_name,
    source_image,
    image_bytes,
    workflow_spec,
    user_api_key,
    image_input_name,
    run_id,
    run_started_at,
):
    """
    Executes inline workflow execution via Roboflow API.

    This keeps the behavior you want:
    each website user enters their own Roboflow API key,
    and their Roboflow account credits are used.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    url = f"https://detect.roboflow.com/infer/workflows?api_key={user_api_key}"

    payload = {
        "specification": workflow_spec,
        "inputs": {
            image_input_name: {
                "type": "base64",
                "value": b64_image,
            }
        },
        "api_key": user_api_key,
    }

    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=180)

            if not response.ok:
                try:
                    err_detail = response.json()
                except Exception:
                    err_detail = response.text

                last_error = f"HTTP {response.status_code}: {err_detail}"

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue

                return build_error_row(
                    run_id=run_id,
                    run_started_at=run_started_at,
                    source_image=source_image,
                    file_name=file_name,
                    status="workflow_error",
                    error=last_error,
                    attempts=attempt,
                )

            data = response.json()
            output = unwrap_workflow_output(data)

            row = {
                "run_id": run_id,
                "run_started_at": run_started_at.isoformat(timespec="seconds"),
                "source_image": source_image,
                "file_name": file_name,

                # Main workflow outputs.
                "status": clean_scalar(output.get("status")),
                "registration_number": clean_scalar(output.get("registration_number")),
                "partial_registration_candidate": clean_scalar(output.get("partial_registration_candidate")),
                "boat_count": clean_scalar(output.get("boat_count")),
                "boat_detected": clean_scalar(output.get("boat_detected")),
                "date_and_time": clean_scalar(output.get("date_and_time")),
                "output_line": clean_scalar(output.get("output_line")),

                # OCR/debug text outputs.
                "raw_ocr_text": clean_scalar(output.get("raw_ocr_text")),
                "all_visible_text": clean_scalar(output.get("all_visible_text")),
                "other_visible_text": clean_scalar(output.get("other_visible_text")),
                "easyocr_visible_text": clean_scalar(output.get("easyocr_visible_text")),
                "verified_text_region_ocr": clean_scalar(output.get("verified_text_region_ocr")),
                "focused_registration_ocr": clean_scalar(output.get("focused_registration_ocr")),

                # Boat classification outputs.
                "boat_type": clean_scalar(output.get("boat_type")),
                "boat_type_identifier_confidence": clean_scalar(output.get("boat_type_identifier_confidence")),
                "approved_identifiers_found": clean_scalar(output.get("approved_identifiers_found")),
                "unapproved_identifiers": clean_scalar(output.get("unapproved_identifiers")),
                "classification_reason": clean_scalar(output.get("classification_reason")),

                # Processing metadata.
                "workflow_error": "",
                "attempts": attempt,
            }

            return row

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    return build_error_row(
        run_id=run_id,
        run_started_at=run_started_at,
        source_image=source_image,
        file_name=file_name,
        status="workflow_error",
        error=last_error,
        attempts=MAX_RETRIES,
    )


def parse_datetime_value(value):
    """Attempts to parse a datetime string safely."""
    text = str(value or "").strip()

    if not text:
        return pd.NaT

    return pd.to_datetime(text, errors="coerce")


def get_row_datetime(row):
    """
    Uses workflow date_and_time when available.
    Falls back to run_started_at when date_and_time is missing.
    """
    workflow_dt = parse_datetime_value(row.get("date_and_time"))

    if pd.notnull(workflow_dt):
        return workflow_dt

    run_dt = parse_datetime_value(row.get("run_started_at"))

    if pd.notnull(run_dt):
        return run_dt

    return pd.NaT


def derive_capture_date(dt_value):
    """Returns capture_date in MM/DD/YYYY style when possible."""
    if pd.isnull(dt_value):
        return ""

    return dt_value.strftime("%m/%d/%Y")


def derive_summary_registration_source(row):
    """
    Creates a summary registration source value.

    If your workflow already outputs summary_registration_source in the future,
    this function will use it. Otherwise, it derives a useful value.
    """
    existing = str(row.get("summary_registration_source") or "").strip()

    if existing:
        return existing

    registration = str(row.get("registration_number") or "").strip()
    partial = str(row.get("partial_registration_candidate") or "").strip()
    status = str(row.get("status") or "").strip().upper()

    if not registration:
        if partial:
            return "PARTIAL_OR_REVIEW"
        return "UNDETECTED"

    if status in ["CONFIRMED", "SUCCESS", "PASSED", "PASS"]:
        return "AUTOMATED_OCR"

    if partial:
        return "PARTIAL_OR_REVIEW"

    return "AUTOMATED_OCR"


def is_confirmed_row(row):
    """Determines whether a row should count as confirmed."""
    status = str(row.get("status") or "").strip().upper()
    registration = str(row.get("registration_number") or "").strip().upper()

    if status == "CONFIRMED":
        return True

    if status in ["SUCCESS", "PASSED", "PASS"] and registration not in ["", "UNDETECTED", "NONE", "UNKNOWN"]:
        return True

    if registration not in ["", "UNDETECTED", "NONE", "UNKNOWN"]:
        if "ERROR" not in status and "FAIL" not in status and "UNDETECTED" not in status:
            return True

    return False


def is_partial_or_review_row(row):
    """Determines whether a row should count as partial or needs review."""
    status = str(row.get("status") or "").strip().upper()
    partial = str(row.get("partial_registration_candidate") or "").strip()

    if "REVIEW" in status or "PARTIAL" in status:
        return True

    if partial:
        registration = str(row.get("registration_number") or "").strip()
        if not registration:
            return True

    return False


def is_undetected_row(row):
    """Determines whether a row should count as undetected."""
    status = str(row.get("status") or "").strip().upper()
    registration = str(row.get("registration_number") or "").strip().upper()

    if registration in ["", "UNDETECTED", "NONE", "UNKNOWN"]:
        return True

    if "UNDETECTED" in status or status in ["NONE", "UNKNOWN"]:
        return True

    return False


def build_daily_summary(df_results):
    """
    Groups detailed results and constructs the requested Daily Summary table.

    Output columns:
    capture_date
    summary_registration
    summary_registration_source
    appearance_count
    first_seen
    last_seen
    trip_length_hhmmss
    trip_length_seconds
    trip_length_status
    confirmed_registration_count
    partial_or_review_count
    undetected_count
    first_image
    last_image
    image_names
    boat_types
    """
    if df_results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df_work = df_results.copy()

    # Ensure all expected detail columns exist.
    for col in DETAIL_COLUMNS:
        if col not in df_work.columns:
            df_work[col] = ""

    df_work["dt_ts"] = df_work.apply(get_row_datetime, axis=1)
    df_work["capture_date"] = df_work["dt_ts"].apply(derive_capture_date)

    df_work["summary_registration"] = (
        df_work["registration_number"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Group blank registrations under UNDETECTED.
    df_work.loc[df_work["summary_registration"] == "", "summary_registration"] = "UNDETECTED"

    df_work["summary_registration_source"] = df_work.apply(derive_summary_registration_source, axis=1)

    summary_rows = []

    grouped = df_work.groupby(
        ["capture_date", "summary_registration"],
        dropna=False,
        sort=True,
    )

    for (capture_date, summary_registration), group in grouped:
        sorted_group = group.sort_values("dt_ts", na_position="last")

        appearance_count = len(sorted_group)

        valid_times = sorted_group["dt_ts"].dropna()

        if not valid_times.empty:
            first_dt = valid_times.min()
            last_dt = valid_times.max()

            first_seen = first_dt.strftime("%H:%M:%S")
            last_seen = last_dt.strftime("%H:%M:%S")

            trip_length_seconds = int((last_dt - first_dt).total_seconds())
            hours, remainder = divmod(trip_length_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            trip_length_hhmmss = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            first_seen = ""
            last_seen = ""
            trip_length_seconds = 0
            trip_length_hhmmss = "00:00:00"

        trip_length_status = "COMPLETED" if appearance_count > 1 else "SINGLE_DETECTION"

        confirmed_registration_count = int(
            sorted_group.apply(is_confirmed_row, axis=1).sum()
        )

        partial_or_review_count = int(
            sorted_group.apply(is_partial_or_review_row, axis=1).sum()
        )

        undetected_count = int(
            sorted_group.apply(is_undetected_row, axis=1).sum()
        )

        first_image = sorted_group["file_name"].iloc[0] if not sorted_group.empty else ""
        last_image = sorted_group["file_name"].iloc[-1] if not sorted_group.empty else ""

        image_names = " | ".join(
            sorted_group["file_name"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        boat_types_seen = []

        for boat_type in sorted_group["boat_type"].fillna("").astype(str).tolist():
            boat_type = boat_type.strip()

            if not boat_type:
                continue

            if boat_type not in boat_types_seen:
                boat_types_seen.append(boat_type)

        boat_types = " | ".join(boat_types_seen)

        source_values = (
            sorted_group["summary_registration_source"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        source_values = [v for v in source_values.tolist() if v]

        summary_registration_source = source_values[0] if source_values else ""

        summary_rows.append(
            {
                "capture_date": capture_date,
                "summary_registration": summary_registration,
                "summary_registration_source": summary_registration_source,
                "appearance_count": appearance_count,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "trip_length_hhmmss": trip_length_hhmmss,
                "trip_length_seconds": trip_length_seconds,
                "trip_length_status": trip_length_status,
                "confirmed_registration_count": confirmed_registration_count,
                "partial_or_review_count": partial_or_review_count,
                "undetected_count": undetected_count,
                "first_image": first_image,
                "last_image": last_image,
                "image_names": image_names,
                "boat_types": boat_types,
            }
        )

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def dataframe_to_excel_bytes(df, sheet_name):
    """Converts a DataFrame to downloadable Excel bytes."""
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    return excel_buffer.getvalue()


def dataframe_to_csv_bytes(df):
    """Converts a DataFrame to downloadable CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


# ==========================================
# 3. SESSION STATE INITIALIZATION
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

if "csv_results_bytes" not in st.session_state:
    st.session_state.csv_results_bytes = None

if "csv_summary_bytes" not in st.session_state:
    st.session_state.csv_summary_bytes = None


# ==========================================
# 4. STREAMLIT UI & CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Boat Registration & Classification System",
    page_icon="🛥️",
    layout="wide",
)

st.title("🛥️ Boat Registration & Classification System")
st.markdown(
    "Upload individual images, dropped folders, or ZIP archives for automated boat registration and classification analysis."
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Roboflow API Key",
        type="password",
        help="Enter your personal Roboflow Private API Key. Runs will use your Roboflow account credits.",
    )

    st.divider()

    max_workers = st.slider(
        "Parallel Processing Threads",
        min_value=1,
        max_value=10,
        value=3,
        help="Keep this modest because the workflow is heavy.",
    )

# File / Folder Dropzone
st.subheader("📁 Input Files")

uploaded_items = st.file_uploader(
    "Drag and drop image files, folders, or ZIP archives here:",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif", "zip"],
    accept_multiple_files=True,
)

server_output_dir = st.text_input(
    "Optional Server Save Path (Leave blank to download via browser):"
)


# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================

if st.button("🚀 Process Batch", type="primary", disabled=not (uploaded_items and api_key)):
    st.session_state.run_complete = False

    run_started_at = datetime.datetime.now()
    run_id = run_started_at.strftime("%Y-%m-%d_%H-%M-%S")

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
                    image_item["file_name"],
                    image_item["source_image"],
                    image_item["image_bytes"],
                    workflow_spec,
                    api_key.strip(),
                    image_input_name,
                    run_id,
                    run_started_at,
                ): image_item["file_name"]
                for image_item in image_batch
            }

            for future in as_completed(future_to_file):
                try:
                    res = future.result()
                except Exception as e:
                    file_name = future_to_file[future]
                    res = build_error_row(
                        run_id=run_id,
                        run_started_at=run_started_at,
                        source_image=file_name,
                        file_name=file_name,
                        status="script_error",
                        error=f"{type(e).__name__}: {e}",
                        attempts=MAX_RETRIES,
                    )

                results.append(res)

                completed_count += 1
                progress_bar.progress(completed_count / total_count)

                st.write(
                    f"[{completed_count}/{total_count}] "
                    f"{res.get('file_name')} | "
                    f"status={res.get('status')} | "
                    f"registration={res.get('registration_number')}"
                )

        status.update(label="Processing complete! Generating output files...", state="complete")

    # Sort output rows in a predictable order.
    results = sorted(results, key=lambda r: r.get("source_image", ""))

    # Generate detailed results DataFrame.
    df_res = pd.DataFrame(results)

    # Force exact detailed column order.
    for col in DETAIL_COLUMNS:
        if col not in df_res.columns:
            df_res[col] = ""

    df_res = df_res[DETAIL_COLUMNS]

    # Generate daily summary DataFrame.
    df_sum = build_daily_summary(df_res)

    # Output 1: Detailed Classification Results Excel.
    excel_res_bytes = dataframe_to_excel_bytes(df_res, "Detailed Results")

    # Output 2: Daily Registration Summary Excel.
    excel_sum_bytes = dataframe_to_excel_bytes(df_sum, "Daily Summary")

    # Output 3: Detailed Classification Results CSV.
    csv_res_bytes = dataframe_to_csv_bytes(df_res)

    # Output 4: Daily Registration Summary CSV.
    csv_sum_bytes = dataframe_to_csv_bytes(df_sum)

    # Optional Server File Persistence.
    if server_output_dir:
        os.makedirs(server_output_dir, exist_ok=True)

        detailed_xlsx_path = os.path.join(
            server_output_dir,
            f"detailed_classification_results_{run_id}.xlsx",
        )

        summary_xlsx_path = os.path.join(
            server_output_dir,
            f"daily_registration_summary_{run_id}.xlsx",
        )

        detailed_csv_path = os.path.join(
            server_output_dir,
            f"roboflow_results_clean_{run_id}.csv",
        )

        summary_csv_path = os.path.join(
            server_output_dir,
            f"registration_daily_summary_{run_id}.csv",
        )

        with open(detailed_xlsx_path, "wb") as f:
            f.write(excel_res_bytes)

        with open(summary_xlsx_path, "wb") as f:
            f.write(excel_sum_bytes)

        with open(detailed_csv_path, "wb") as f:
            f.write(csv_res_bytes)

        with open(summary_csv_path, "wb") as f:
            f.write(csv_sum_bytes)

        st.success(f"Excel and CSV files saved to server at: `{server_output_dir}`")

    # Session State Updates.
    st.session_state.df_results = df_res
    st.session_state.df_summary = df_sum
    st.session_state.excel_results_bytes = excel_res_bytes
    st.session_state.excel_summary_bytes = excel_sum_bytes
    st.session_state.csv_results_bytes = csv_res_bytes
    st.session_state.csv_summary_bytes = csv_sum_bytes
    st.session_state.run_complete = True


# ==========================================
# 6. PERSISTENT DISPLAY & DOWNLOAD SECTION
# ==========================================

if st.session_state.run_complete:
    st.divider()
    st.header("📊 Results & Generated Files")

    tab1, tab2 = st.tabs(
        [
            "📄 Detailed Classification Data",
            "📅 Daily Registration Summary",
        ]
    )

    with tab1:
        st.dataframe(st.session_state.df_results, use_container_width=True)

    with tab2:
        st.dataframe(st.session_state.df_summary, use_container_width=True)

    st.subheader("📥 Download Output Files")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📘 Download Detailed Results (.xlsx)",
            data=st.session_state.excel_results_bytes,
            file_name="detailed_classification_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.download_button(
            label="📄 Download Detailed Results (.csv)",
            data=st.session_state.csv_results_bytes,
            file_name="roboflow_results_clean.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="📊 Download Daily Registration Summary (.xlsx)",
            data=st.session_state.excel_summary_bytes,
            file_name="daily_registration_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.download_button(
            label="📅 Download Daily Registration Summary (.csv)",
            data=st.session_state.csv_summary_bytes,
            file_name="registration_daily_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
