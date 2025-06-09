import logging
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, zipfile, shutil
from tempfile import TemporaryDirectory
from ..services.ecg_processing_service import ECGProcessingService
from ..services.file_service import FileHandlingService
from ..services.prediction_service import PredictionService
from ..services.storage_service import StorageService
from ..models import db, ECG, generate_ecg_file_id
import requests


ecg_bp = Blueprint("ecg", __name__)
logger = logging.getLogger(__name__)

def validate_upload_file(file):
    if file is None:
        logger.warning("File part missing in upload")
        raise ValueError("No file part")
    if not file.filename.lower().endswith(".zip"):
        logger.warning(f"Invalid file extension: {file.filename}")
        raise ValueError("ZIP required")


def save_and_extract_zip(file, file_id):
    temp_path = f"/tmp/{file_id}.zip"
    file.save(temp_path)

    extract_dir = f"/tmp/{file_id}"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(temp_path, "r") as z:
        z.extractall(extract_dir)
    os.remove(temp_path)

    return extract_dir


def convert_edf_files(rec_dir):
    for filename in os.listdir(rec_dir):
        if filename.lower().endswith(".edf"):
            edf_path = os.path.join(rec_dir, filename)
            FileHandlingService.convert_edf_to_wfdb(edf_path, output_dir=rec_dir)
            os.remove(edf_path)
            return  # Assuming one EDF file max


def ensure_hea_exists(rec_dir):
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith(".hea")), None)
    if not hea:
        raise FileNotFoundError("No .hea found")
    return hea


def clean_non_wfdb_files(rec_dir):
    allowed_exts = {".hea", ".dat", ".atr"}
    for file in os.listdir(rec_dir):
        if os.path.splitext(file)[1].lower() not in allowed_exts:
            os.remove(os.path.join(rec_dir, file))


def create_wfdb_zip(rec_dir, file_id):
    zip_path = f"/tmp/{file_id}_wfdb.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in os.listdir(rec_dir):
            zipf.write(os.path.join(rec_dir, f), arcname=f)
    return zip_path


def upload_zip_to_storage(zip_path, file_id):
    storage_path = f"{file_id}/wfdb.zip"
    file_url = StorageService.upload_file_to_supabase(zip_path, storage_path)
    return storage_path, file_url


def analyze_ecg_and_plot(wfdb_basename, file_id):
    plot_folder = "/tmp/plots"
    os.makedirs(plot_folder, exist_ok=True)
    return ECGProcessingService.analyze_and_plot(
        wfdb_basename=wfdb_basename, plot_folder=plot_folder, file_id=file_id
    )


def upload_plot(plot_path, file_id):
    plot_storage_path = f"{file_id}/plot.png"
    plot_public_url = StorageService.upload_file_to_supabase(plot_path, plot_storage_path)
    return plot_storage_path, plot_public_url


def extract_metadata(form):
    patient_name = form.get("patient_name")
    age = form.get("age")
    age = int(age) if age and age.isdigit() else None
    gender = form.get("gender")
    notes = form.get("notes")
    return patient_name, age, gender, notes


def predict_summary(summary, age, gender):
    return PredictionService.predict_ecg_classification(summary, age=age, gender=gender)


def save_ecg_record(
    file_id, user_id, patient_name, age, gender, summary, notes, wfdb_path, plot_path
):
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


def cleanup_files(*paths):
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass


@ecg_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict_wfdb():
    user_id = int(get_jwt_identity())
    file_id = generate_ecg_file_id()
    logger.info("ECG predict request received", extra={"user_id": user_id, "file_id": file_id})

    try:
        file = request.files.get("file")
        validate_upload_file(file)
        logger.info("File validated successfully", extra={"file_id": file_id})

        rec_dir = save_and_extract_zip(file, file_id)
        logger.info("Zip file saved and extracted", extra={"rec_dir": rec_dir, "file_id": file_id})

        convert_edf_files(rec_dir)
        logger.info("EDF files converted", extra={"rec_dir": rec_dir, "file_id": file_id})

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

        patient_name, age, gender, notes = extract_metadata(request.form)
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

        cleanup_files(rec_dir, wfdb_zip_path, plot_path)
        logger.info("Temporary files cleaned up", extra={"file_id": file_id})

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

    except Exception as e:
        logger.exception("WFDB analysis failed on predict", exc_info=e)
        cleanup_files(
            rec_dir if "rec_dir" in locals() else None,
            wfdb_zip_path if "wfdb_zip_path" in locals() else None,
            plot_path if "plot_path" in locals() else None,
        )
        return jsonify({"error": str(e)}), 400


@ecg_bp.route("/ecg/<file_id>", methods=["GET"])
@jwt_required()
def get_record(file_id):
    user_id = int(get_jwt_identity())
    logger.info("Get record request", extra={"user_id": user_id, "file_id": file_id})

    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        logger.warning("Record not found", extra={"file_id": file_id, "user_id": user_id})
        return jsonify({"error": "Record not found"}), 404

    plot_url = StorageService.generate_signed_url(ecg.plot_path)
    record_data = ecg.to_dict()
    record_data["plot_url"] = plot_url

    logger.info("Record data retrieved successfully", extra={"file_id": file_id, "user_id": user_id})
    return jsonify(record_data), 200


@ecg_bp.route("/history", methods=["GET"])
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    search = request.args.get("search", "").lower()
    logger.info("History request", extra={"user_id": user_id, "search": search})

    query = ECG.query.filter_by(user_id=user_id)
    if search:
        query = query.filter(ECG.patient_name.ilike(f"%{search}%"))

    records = query.order_by(ECG.upload_date.desc()).all()
    logger.info(f"Found {len(records)} records", extra={"user_id": user_id})
    return jsonify([record.to_dict() for record in records]), 200


@ecg_bp.route("/ecg/<file_id>/leads", methods=["GET"])
@jwt_required()
def get_ecg_leads(file_id):
    user_id = int(get_jwt_identity())
    logger.info("Get ECG leads request", extra={"user_id": user_id, "file_id": file_id})

    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        logger.warning("Record not found", extra={"file_id": file_id, "user_id": user_id})
        return jsonify({"error": "Record not found"}), 404

    try:
        zip_url = StorageService.generate_signed_url(ecg.wfdb_path)
        with TemporaryDirectory() as temp_dir:
            local_zip_path = os.path.join(temp_dir, "ecg.zip")
            resp = requests.get(zip_url)
            resp.raise_for_status()
            with open(local_zip_path, "wb") as f:
                f.write(resp.content)

            with zipfile.ZipFile(local_zip_path, "r") as z:
                z.extractall(temp_dir)

            data = FileHandlingService.load_and_clean_all_leads(temp_dir)

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

    except requests.HTTPError as e:
        logger.error(f"Failed to download ECG zip: {e}", exc_info=e)
        return jsonify({"error": "Failed to download ECG zip file"}), 500
    except FileNotFoundError as e:
        logger.warning(f"File not found during leads extraction: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception(f"Failed to load ECG leads for {file_id}", exc_info=e)
        return jsonify({"error": f"Failed to load ECG leads: {e}"}), 500


@ecg_bp.route("/ecg/<file_id>/notes", methods=["PUT"])
@jwt_required()
def update_ecg_notes(file_id):
    user_id = int(get_jwt_identity())
    logger.info("Update ECG notes request", extra={"user_id": user_id, "file_id": file_id})

    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        logger.warning("Record not found for notes update", extra={"file_id": file_id, "user_id": user_id})
        return jsonify({"error": "Record not found"}), 404

    data = request.get_json()
    if not data or "notes" not in data:
        logger.warning("Missing notes in request body", extra={"user_id": user_id, "file_id": file_id})
        return jsonify({"error": "Missing notes in request body"}), 400

    try:
        ecg.notes = data["notes"]
        db.session.commit()
        logger.info("Notes updated successfully", extra={"file_id": file_id, "user_id": user_id})
        return jsonify({"message": "Notes updated successfully", "notes": ecg.notes}), 200
    except Exception as e:
        logger.exception(f"Failed to update notes for {file_id}", exc_info=e)
        db.session.rollback()
        return jsonify({"error": f"Failed to update notes: {str(e)}"}), 500
