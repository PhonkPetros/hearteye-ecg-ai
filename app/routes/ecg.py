import logging
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, zipfile, shutil
from tempfile import TemporaryDirectory
import requests
from marshmallow import ValidationError
from ..utils.error_handlers import handle_errors
from ..utils.exceptions import APIError, ErrorCodes
from ..services.ecg_processing_service import ECGProcessingService
from ..services.file_service import FileHandlingService
from ..services.prediction_service import PredictionService
from ..services.storage_service import StorageService
from ..models import db, ECG, generate_ecg_file_id
from ..schemas import ECGUploadSchema


ecg_bp = Blueprint("ecg", __name__)
logger = logging.getLogger(__name__)

def validate_upload_file(file):
    if file is None:
        logger.warning("File part missing in upload")
        raise APIError(ErrorCodes.MISSING_FIELD, "No file part in the request.", status_code=400, details={"field": "file"})
    if not file.filename or not file.filename.lower().endswith(".zip"):
        logger.warning(f"Invalid file extension: {file.filename}")
        raise APIError(ErrorCodes.INVALID_FILE_TYPE, "Only ZIP files are allowed.", status_code=400, details={"filename": file.filename})

def save_and_extract_zip(file, file_id):
    temp_path = f"/tmp/{file_id}.zip"
    file.save(temp_path)

    extract_dir = f"/tmp/{file_id}"
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(temp_path, "r") as z:
            z.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise APIError(ErrorCodes.ECG_CORRUPTED, "Uploaded file is a corrupted or invalid ZIP archive.", status_code=400)
    finally:
        os.remove(temp_path)
    return extract_dir

def convert_edf_files(rec_dir):
    converted = False
    for filename in os.listdir(rec_dir):
        if filename.lower().endswith(".edf"):
            edf_path = os.path.join(rec_dir, filename)
            try:
                FileHandlingService.convert_edf_to_wfdb(edf_path, output_dir=rec_dir)
                os.remove(edf_path)
                converted = True
                break
            except Exception as e:
                raise APIError(ErrorCodes.ECG_PROCESSING_FAILED, f"Failed to convert EDF file: {str(e)}", status_code=500, details={"file": filename})
    return converted

def ensure_hea_exists(rec_dir):
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith(".hea")), None)
    if not hea:
        raise APIError(ErrorCodes.ECG_INSUFFICIENT_DATA, "No .hea file found in the uploaded ECG data. A header file is required for WFDB processing.", status_code=400)
    return hea

def clean_non_wfdb_files(rec_dir):
    allowed_exts = {".hea", ".dat", ".atr"}
    for file in os.listdir(rec_dir):
        if os.path.splitext(file)[1].lower() not in allowed_exts:
            try:
                os.remove(os.path.join(rec_dir, file))
            except OSError as e:
                logger.warning(f"Could not remove non-WFDB file {file}: {e}")

def create_wfdb_zip(rec_dir, file_id):
    zip_path = f"/tmp/{file_id}_wfdb.zip"
    try:
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for f in os.listdir(rec_dir):
                zipf.write(os.path.join(rec_dir, f), arcname=f)
    except Exception as e:
        raise APIError(ErrorCodes.INTERNAL_ERROR, f"Failed to create WFDB zip: {str(e)}", status_code=500)
    return zip_path

def upload_zip_to_storage(zip_path, file_id):
    storage_path = f"{file_id}/wfdb.zip"
    try:
        file_url = StorageService.upload_file_to_supabase(zip_path, storage_path)
        return storage_path, file_url
    except Exception as e:
        raise APIError(ErrorCodes.SERVICE_UNAVAILABLE, f"Failed to upload WFDB zip to storage: {str(e)}", status_code=500)

def analyze_ecg_and_plot(wfdb_basename, file_id):
    plot_folder = "/tmp/plots"
    os.makedirs(plot_folder, exist_ok=True)
    try:
        summary, plot_path = ECGProcessingService.analyze_and_plot(
            wfdb_basename=wfdb_basename, plot_folder=plot_folder, file_id=file_id
        )
        if not summary:
            raise APIError(ErrorCodes.ECG_PROCESSING_FAILED, "ECG analysis produced no summary data.", status_code=500)
        return summary, plot_path
    except Exception as e:
        raise APIError(ErrorCodes.ECG_PROCESSING_FAILED, f"ECG analysis failed: {str(e)}", status_code=500)

def upload_plot(plot_path, file_id):
    plot_storage_path = f"{file_id}/plot.png"
    try:
        plot_public_url = StorageService.upload_file_to_supabase(plot_path, plot_storage_path)
        return plot_storage_path, plot_public_url
    except Exception as e:
        raise APIError(ErrorCodes.SERVICE_UNAVAILABLE, f"Failed to upload plot to storage: {str(e)}", status_code=500)

def extract_and_validate_metadata(form):
    schema = ECGUploadSchema()
    try:
        validated_data = schema.load(form)
        return validated_data["patient_name"], validated_data["age"], validated_data["gender"], validated_data.get("notes")
    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error during metadata extraction/validation", exc_info=e)
        raise APIError(ErrorCodes.INTERNAL_ERROR, "Error processing patient metadata.", status_code=500)

def predict_summary(summary, age, gender):
    try:
        return PredictionService.predict_ecg_classification(summary, age=age, gender=gender)
    except Exception as e:
        raise APIError(ErrorCodes.INTERNAL_ERROR, f"Prediction service failed: {str(e)}", status_code=500)

def save_ecg_record(
    file_id, user_id, patient_name, age, gender, summary, notes, wfdb_path, plot_path
):
    try:
        ecg = ECG(
            file_id=file_id,
            user_id=user_id,
            patient_name=patient_name,
            age=age,
            gender=gender,
            heart_rate=summary.get("heart_rate"),
            p_wave_duration=summary.get("intervals", {}).get("P_wave_duration_ms"),
            pq_interval=summary.get("intervals", {}).get("PQ_interval_ms"),
            qrs_duration=summary.get("intervals", {}).get("QRS_duration_ms"),
            qt_interval=summary.get("intervals", {}).get("QT_interval_ms"),
            classification=summary.get("classification"),
            confidence=summary.get("confidence"),
            notes=notes,
            wfdb_path=wfdb_path,
            plot_path=plot_path,
        )
        db.session.add(ecg)
        db.session.commit()
        return ecg
    except Exception as e:
        db.session.rollback()
        raise APIError(ErrorCodes.INTERNAL_ERROR, f"Failed to save ECG record to database: {str(e)}", status_code=500)

def cleanup_files(*paths):
    for path in paths:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Could not remove temporary file {path}: {e}")

@ecg_bp.route("/predict", methods=["POST"])
@jwt_required()
@handle_errors 
def predict_wfdb():
    user_id = int(get_jwt_identity())
    file_id = generate_ecg_file_id()
    logger.info("ECG predict request received", extra={"user_id": user_id, "file_id": file_id})

    # Initialize variables for cleanup_files in case of early error
    rec_dir = None
    wfdb_zip_path = None
    plot_path = None

    try:
        file = request.files.get("file")
        validate_upload_file(file)
        logger.info("File validated successfully", extra={"file_id": file_id})

        rec_dir = save_and_extract_zip(file, file_id)
        logger.info("Zip file saved and extracted", extra={"rec_dir": rec_dir, "file_id": file_id})

        converted = convert_edf_files(rec_dir)
        if converted:
            logger.info("EDF files converted", extra={"rec_dir": rec_dir, "file_id": file_id})
        else:
            logger.info("No EDF files to convert or conversion not needed", extra={"rec_dir": rec_dir, "file_id": file_id})

        hea_file = ensure_hea_exists(rec_dir)
        logger.info(".hea file found", extra={"hea_file": hea_file, "file_id": file_id})

        clean_non_wfdb_files(rec_dir)
        logger.info("Cleaned non-WFDB files", extra={"rec_dir": rec_dir, "file_id": file_id})

        wfdb_zip_path = create_wfdb_zip(rec_dir, file_id)
        wfdb_storage_path, _ = upload_zip_to_storage(wfdb_zip_path, file_id)
        logger.info("WFDB zip uploaded to storage", extra={"wfdb_storage_path": wfdb_storage_path, "file_id": file_id})

        wfdb_basename = os.path.join(rec_dir, os.path.splitext(hea_file)[0])
        summary, plot_path = analyze_ecg_and_plot(wfdb_basename, file_id)
        logger.info("ECG analyzed and plot generated", extra={"file_id": file_id})

        plot_storage_path, plot_public_url = upload_plot(plot_path, file_id)
        logger.info("Plot uploaded to storage", extra={"plot_storage_path": plot_storage_path, "file_id": file_id})

        patient_name, age, gender, notes = extract_and_validate_metadata(request.form)
        summary = predict_summary(summary, age, gender)
        logger.info("Summary prediction done", extra={"summary": summary, "file_id": file_id})

        ecg_record = save_ecg_record(
            file_id,
            user_id,
            patient_name,
            age,
            gender,
            summary,
            notes,
            wfdb_storage_path,
            plot_storage_path,
        )
        logger.info("ECG record saved in DB", extra={"ecg_record_id": ecg_record.id, "file_id": file_id})

        return (
            jsonify(
                {
                    "file_id": file_id,
                    "summary": summary,
                    "plot": plot_public_url,
                    "record": ecg_record.to_dict(),
                }
            ),
            201,
        )
    finally:
        cleanup_files(rec_dir, wfdb_zip_path, plot_path)
        logger.info("Temporary files cleaned up", extra={"file_id": file_id})


@ecg_bp.route("/ecg/<file_id>", methods=["GET"])
@jwt_required()
@handle_errors 
def get_record(file_id):
    user_id = int(get_jwt_identity())
    logger.info("Get record request", extra={"user_id": user_id, "file_id": file_id})

    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        logger.warning("Record not found", extra={"file_id": file_id, "user_id": user_id})
        raise APIError(ErrorCodes.RECORD_NOT_FOUND, "ECG record not found.", status_code=404)

    try:
        plot_url = StorageService.generate_signed_url(ecg.plot_path)
    except Exception as e:
        logger.error(f"Failed to generate signed URL for plot: {e}", exc_info=e)
        raise APIError(ErrorCodes.SERVICE_UNAVAILABLE, "Failed to retrieve plot URL.", status_code=500)

    record_data = ecg.to_dict()
    record_data["plot_url"] = plot_url

    logger.info("Record data retrieved successfully", extra={"file_id": file_id, "user_id": user_id})
    return jsonify(record_data), 200


@ecg_bp.route("/history", methods=["GET"])
@jwt_required()
@handle_errors 
def history():
    user_id = int(get_jwt_identity())
    search = request.args.get("search", "").lower()
    logger.info("History request", extra={"user_id": user_id, "search": search})

    try:
        query = ECG.query.filter_by(user_id=user_id)
        if search:
            query = query.filter(ECG.patient_name.ilike(f"%{search}%"))

        records = query.order_by(ECG.upload_date.desc()).all()
        logger.info(f"Found {len(records)} records", extra={"user_id": user_id})
        return jsonify([record.to_dict() for record in records]), 200
    except Exception as e:
        logger.exception("Failed to retrieve ECG history", exc_info=e)
        raise APIError(ErrorCodes.INTERNAL_ERROR, "Failed to retrieve ECG history.", status_code=500)


@ecg_bp.route("/ecg/<file_id>/leads", methods=["GET"])
@jwt_required()
@handle_errors 
def get_ecg_leads(file_id):
    user_id = int(get_jwt_identity())
    logger.info("Get ECG leads request", extra={"user_id": user_id, "file_id": file_id})

    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        logger.warning("Record not found", extra={"file_id": file_id, "user_id": user_id})
        raise APIError(ErrorCodes.RECORD_NOT_FOUND, "ECG record not found.", status_code=404)

    temp_dir = None
    try:
        zip_url = StorageService.generate_signed_url(ecg.wfdb_path)
        temp_dir = TemporaryDirectory()
        local_zip_path = os.path.join(temp_dir.name, "ecg.zip")
        resp = requests.get(zip_url)
        resp.raise_for_status() # This will raise an HTTPError for bad responses
        with open(local_zip_path, "wb") as f:
            f.write(resp.content)

        with zipfile.ZipFile(local_zip_path, "r") as z:
            z.extractall(temp_dir.name)

        data = FileHandlingService.load_and_clean_all_leads(temp_dir.name)

        logger.info("ECG leads data loaded", extra={"file_id": file_id, "user_id": user_id})
        return jsonify(
            {
                "fs": data["fs"],
                "leads": data["lead_names"],
                "signals": data["cleaned_signals"].T.tolist(),
                "patient_name": ecg.patient_name,
                "age": ecg.age,
                "gender": ecg.gender,
                "upload_date": ecg.upload_date,
                "p_wave_duration": ecg.p_wave_duration,
                "pq_interval": ecg.pq_interval,
                "qrs_duration": ecg.qrs_duration,
                "qt_interval": ecg.qt_interval,
                "classification": ecg.classification,
                "confidence": ecg.confidence,
                "notes": ecg.notes,
            }
        ), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download ECG zip from storage: {e}", exc_info=e)
        raise APIError(ErrorCodes.SERVICE_UNAVAILABLE, "Failed to download ECG data from storage.", status_code=500)
    except zipfile.BadZipFile:
        logger.error(f"Downloaded WFDB zip is corrupted for file_id: {file_id}")
        raise APIError(ErrorCodes.ECG_CORRUPTED, "Downloaded ECG data is corrupted.", status_code=500)
    except FileNotFoundError as e:
        logger.warning(f"File not found during leads extraction: {e}", extra={"file_id": file_id})
        raise APIError(ErrorCodes.RECORD_NOT_FOUND, f"Required file for leads extraction not found: {str(e)}", status_code=404)
    except Exception as e:
        logger.exception(f"Failed to load ECG leads for {file_id}", exc_info=e)
        raise APIError(ErrorCodes.INTERNAL_ERROR, f"Failed to process ECG leads: {str(e)}", status_code=500)
    finally:
        if temp_dir:
            temp_dir.cleanup()
            logger.info("Temporary directory for leads cleanup done", extra={"file_id": file_id})


@ecg_bp.route("/ecg/<file_id>/notes", methods=["PUT"])
@jwt_required()
@handle_errors 
def update_ecg_notes(file_id):
    user_id = int(get_jwt_identity())
    logger.info("Update ECG notes request", extra={"user_id": user_id, "file_id": file_id})

    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        logger.warning("Record not found for notes update", extra={"file_id": file_id, "user_id": user_id})
        raise APIError(ErrorCodes.RECORD_NOT_FOUND, "ECG record not found.", status_code=404)

    data = request.get_json()
    if not data or "notes" not in data:
        logger.warning("Missing notes in request body", extra={"user_id": user_id, "file_id": file_id})
        raise APIError(ErrorCodes.MISSING_FIELD, "Missing 'notes' in request body.", status_code=400, details={"field": "notes"})

    try:
        ecg.notes = data["notes"]
        db.session.commit()
        logger.info("Notes updated successfully", extra={"file_id": file_id, "user_id": user_id})
        return jsonify({"message": "Notes updated successfully", "notes": ecg.notes}), 200
    except Exception as e:
        logger.exception(f"Failed to update notes for {file_id}", exc_info=e)
        db.session.rollback()
        raise APIError(ErrorCodes.INTERNAL_ERROR, f"Failed to update notes: {str(e)}", status_code=500)