from flask import Blueprint, request, jsonify, send_from_directory, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, uuid, zipfile, shutil, logging
from datetime import datetime
from tempfile import TemporaryDirectory
from ..utils import analyze_and_plot, load_and_clean_all_leads, upload_file_to_supabase, generate_signed_url_from_supabase, convert_edf_to_wfdb, predict_ecg_classification
from ..models import db, ECG, User, generate_ecg_file_id
import requests


wfdb_bp = Blueprint('wfdb', __name__)
logger = logging.getLogger(__name__)

def validate_upload_file(file):
    if file is None:
        raise ValueError('No file part')
    if not file.filename.lower().endswith('.zip'):
        raise ValueError('ZIP required')

def save_and_extract_zip(file, file_id):
    temp_path = f"/tmp/{file_id}.zip"
    file.save(temp_path)

    extract_dir = f"/tmp/{file_id}"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(temp_path, 'r') as z:
        z.extractall(extract_dir)
    os.remove(temp_path)

    return extract_dir

def convert_edf_files(rec_dir):
    for filename in os.listdir(rec_dir):
        if filename.lower().endswith('.edf'):
            edf_path = os.path.join(rec_dir, filename)
            convert_edf_to_wfdb(edf_path, output_dir=rec_dir)
            os.remove(edf_path)
            return  # Assuming one EDF file max

def ensure_hea_exists(rec_dir):
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        raise FileNotFoundError('No .hea found')
    return hea

def clean_non_wfdb_files(rec_dir):
    allowed_exts = {'.hea', '.dat', '.atr'}
    for file in os.listdir(rec_dir):
        if os.path.splitext(file)[1].lower() not in allowed_exts:
            os.remove(os.path.join(rec_dir, file))

def create_wfdb_zip(rec_dir, file_id):
    zip_path = f"/tmp/{file_id}_wfdb.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in os.listdir(rec_dir):
            zipf.write(os.path.join(rec_dir, f), arcname=f)
    return zip_path

def upload_zip_to_storage(zip_path, file_id):
    storage_path = f"{file_id}/wfdb.zip"
    file_url = upload_file_to_supabase(zip_path, storage_path)
    return storage_path, file_url

def analyze_ecg_and_plot(wfdb_basename, file_id):
    plot_folder = "/tmp/plots"
    os.makedirs(plot_folder, exist_ok=True)
    return analyze_and_plot(wfdb_basename=wfdb_basename, plot_folder=plot_folder, file_id=file_id)

def upload_plot(plot_path, file_id):
    plot_storage_path = f"{file_id}/plot.png"
    plot_public_url = upload_file_to_supabase(plot_path, plot_storage_path)
    return plot_storage_path, plot_public_url

def extract_metadata(form):
    patient_name = form.get('patient_name')
    age = form.get('age')
    age = int(age) if age and age.isdigit() else None
    gender = form.get('gender')
    notes = form.get('notes')
    return patient_name, age, gender, notes

def predict_summary(summary, age, gender):
    return predict_ecg_classification(summary, age=age, gender=gender)

def save_ecg_record(file_id, user_id, patient_name, age, gender, summary, notes, wfdb_path, plot_path):
    ecg = ECG(
        file_id=file_id,
        user_id=user_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        heart_rate=summary.get('heart_rate'),
        p_wave_duration=summary.get('intervals', {}).get('P_wave_duration_ms'),
        pq_interval=summary.get('intervals', {}).get('PQ_interval_ms'),
        qrs_duration=summary.get('intervals', {}).get('QRS_duration_ms'),
        qt_interval=summary.get('intervals', {}).get('QT_interval_ms'),
        classification=summary.get('classification'),
        confidence=summary.get('confidence'),
        notes=notes,
        wfdb_path=wfdb_path,
        plot_path=plot_path
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

@wfdb_bp.route('/predict', methods=['POST'])
@jwt_required()
def predict_wfdb():
    user_id = int(get_jwt_identity())
    file_id = generate_ecg_file_id()

    try:
        file = request.files.get('file')
        validate_upload_file(file)

        rec_dir = save_and_extract_zip(file, file_id)
        convert_edf_files(rec_dir)

        hea_file = ensure_hea_exists(rec_dir)
        clean_non_wfdb_files(rec_dir)

        wfdb_zip_path = create_wfdb_zip(rec_dir, file_id)
        wfdb_storage_path, _ = upload_zip_to_storage(wfdb_zip_path, file_id)

        wfdb_basename = os.path.join(rec_dir, os.path.splitext(hea_file)[0])
        summary, plot_path = analyze_ecg_and_plot(wfdb_basename, file_id)

        plot_storage_path, plot_public_url = upload_plot(plot_path, file_id)

        patient_name, age, gender, notes = extract_metadata(request.form)
        summary = predict_summary(summary, age, gender)

        ecg_record = save_ecg_record(
            file_id, user_id, patient_name, age, gender,
            summary, notes, wfdb_storage_path, plot_storage_path
        )

        cleanup_files(rec_dir, wfdb_zip_path, plot_path)

        return jsonify({
            'file_id': file_id,
            'summary': summary,
            'plot': plot_public_url,
            'record': ecg_record.to_dict()
        }), 201

    except Exception as e:
        logger.exception('WFDB analysis failed on predict')
        cleanup_files(rec_dir if 'rec_dir' in locals() else None,
                      wfdb_zip_path if 'wfdb_zip_path' in locals() else None,
                      plot_path if 'plot_path' in locals() else None)
        return jsonify({'error': str(e)}), 400


@wfdb_bp.route('/ecg/<file_id>', methods=['GET'])
@jwt_required()
def get_record(file_id):
    user_id = int(get_jwt_identity())
    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    
    if not ecg:
        return jsonify({'error': 'Record not found'}), 404
    
    # Get plot URL
    plot_url = generate_signed_url_from_supabase(ecg.plot_path)
    
    # Get record data
    record_data = ecg.to_dict()
    record_data['plot_url'] = plot_url
    
    return jsonify(record_data), 200

@wfdb_bp.route('/history', methods=['GET'])
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    search = request.args.get('search', '').lower()
    
    # Query records from database
    query = ECG.query.filter_by(user_id=user_id)
    if search:
        query = query.filter(ECG.patient_name.ilike(f'%{search}%'))
    
    records = query.order_by(ECG.upload_date.desc()).all()
    
    return jsonify([record.to_dict() for record in records]), 200

@wfdb_bp.route('/ecg/<file_id>/leads', methods=['GET'])
@jwt_required()
def get_ecg_leads(file_id):
    user_id = int(get_jwt_identity())
    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    if not ecg:
        return jsonify({'error': 'Record not found'}), 404

    try:
        zip_url = generate_signed_url_from_supabase(ecg.wfdb_path)
        with TemporaryDirectory() as temp_dir:
            local_zip_path = os.path.join(temp_dir, "ecg.zip")
            resp = requests.get(zip_url)
            resp.raise_for_status()
            with open(local_zip_path, "wb") as f:
                f.write(resp.content)

            with zipfile.ZipFile(local_zip_path, 'r') as z:
                z.extractall(temp_dir)

            data = load_and_clean_all_leads(temp_dir)

        return jsonify({
            'fs': data['fs'],
            'leads': data['lead_names'],
            'signals': data['cleaned_signals'].T.tolist(),
            'patient_name': ecg.patient_name,
            'age': ecg.age,
            'gender': ecg.gender,
            'upload_date': ecg.upload_date,
            'p_wave_duration': ecg.p_wave_duration,
            'pq_interval': ecg.pq_interval,
            'qrs_duration': ecg.qrs_duration,
            'qt_interval': ecg.qt_interval,
            'classification': ecg.classification,
            'confidence': ecg.confidence,
            'notes': ecg.notes,
        }), 200

    except requests.HTTPError as e:
        logger.error(f"Failed to download ECG zip: {e}")
        return jsonify({'error': 'Failed to download ECG zip file'}), 500
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.exception(f"Failed to load ECG leads for {file_id}")
        return jsonify({'error': f"Failed to load ECG leads: {e}"}), 500

@wfdb_bp.route('/ecg/<file_id>/notes', methods=['PUT'])
@jwt_required()
def update_ecg_notes(file_id):
    user_id = int(get_jwt_identity())
    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()

    if not ecg:
        return jsonify({'error': 'Record not found'}), 404

    data = request.get_json()
    if not data or 'notes' not in data:
        return jsonify({'error': 'Missing notes in request body'}), 400

    try:
        ecg.notes = data['notes']
        db.session.commit()
        return jsonify({'message': 'Notes updated successfully', 'notes': ecg.notes}), 200
    except Exception as e:
        current_app.logger.exception(f"Failed to update notes for {file_id}")
        db.session.rollback()
        return jsonify({'error': f'Failed to update notes: {str(e)}'}), 500