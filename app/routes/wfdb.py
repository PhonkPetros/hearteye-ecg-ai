from flask import Blueprint, request, jsonify, send_from_directory, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, uuid, zipfile, shutil, logging
from datetime import datetime
from tempfile import TemporaryDirectory
from ..utils import analyze_and_plot, load_and_clean_all_leads, upload_file_to_supabase, generate_signed_url_from_supabase
from ..models import db, ECG, User, generate_ecg_file_id
import requests


wfdb_bp = Blueprint('wfdb', __name__)
logger = logging.getLogger(__name__)

@wfdb_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_wfdb():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.zip'):
        return jsonify({'error': 'ZIP required'}), 400
    
    # Get user ID from JWT token
    user_id = int(get_jwt_identity())
    file_id = generate_ecg_file_id()

    #Save locally temporarily
    temp_path = f"/tmp/{file_id}.zip"
    f.save(temp_path)

    #Upload zip to Supabase Storage
    storage_path = f"{file_id}/original.zip"
    file_url = upload_file_to_supabase(temp_path, storage_path)

   # Extract zip locally for analysis
    rec_dir = f"/tmp/{file_id}"
    os.makedirs(rec_dir, exist_ok=True)
    with zipfile.ZipFile(temp_path, 'r') as z:
        z.extractall(rec_dir)

    # Cleanup temp zip if you want
    os.remove(temp_path)

    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        shutil.rmtree(rec_dir, ignore_errors=True)
        return jsonify({'error': 'No .hea found'}), 400

    wfdb_basename = os.path.join(rec_dir, os.path.splitext(hea)[0])
    try:
        plot_folder = "/tmp/plots"
        os.makedirs(plot_folder, exist_ok=True)

        summary, plot_path = analyze_and_plot(
            wfdb_basename=wfdb_basename,
            plot_folder=plot_folder,
            file_id=file_id
        )

        plot_storage_path = f"{file_id}/plot.png"
        plot_public_url = upload_file_to_supabase(plot_path, plot_storage_path)

        # Handle form data with type conversion and None fallback
        patient_name = request.form.get('patient_name')
        age = request.form.get('age')
        age = int(age) if age and age.isdigit() else None
        gender = request.form.get('gender')

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
            notes=request.form.get('notes'),
            wfdb_path=storage_path,
            plot_path=plot_storage_path
        )

        db.session.add(ecg)
        db.session.commit()

        shutil.rmtree(rec_dir, ignore_errors=True)
        os.remove(plot_path)
        return jsonify({
            'file_id': file_id,
            'summary': summary,
            'plot': plot_public_url,
            'record': ecg.to_dict()
        }), 201
        
    except Exception as e:
        logger.exception('WFDB analysis failed on upload')
        shutil.rmtree(rec_dir, ignore_errors=True)
        if os.path.exists(plot_path):
            os.remove(plot_path)
        return jsonify({'error': f'Analysis failed: {e}'}), 500

@wfdb_bp.route('/record/<file_id>', methods=['GET'])
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

@wfdb_bp.route('/analyze/<file_id>', methods=['GET'])
@jwt_required()
def analyze_wfdb(file_id):
    temp_dir = tempfile.mkdtemp()
    plot_folder = None
    try:
        local_record_dir = os.path.join(temp_dir, file_id)
        os.makedirs(local_record_dir, exist_ok=True)

        success = download_supabase_folder(bucket=SUPABASE_BUCKET, 
                                           folder=f"{WFDB_FOLDER}/{file_id}", 
                                           local_path=local_record_dir)
        if not success:
            return jsonify({'error': 'Record not found in storage'}), 404

        hea = next((fn for fn in os.listdir(local_record_dir) if fn.lower().endswith('.hea')), None)
        if not hea:
            return jsonify({'error': 'No .hea found'}), 400

        wfdb_basename = os.path.join(local_record_dir, os.path.splitext(hea)[0])

        plot_folder = tempfile.mkdtemp()
        summary, plot_path = analyze_and_plot(
            wfdb_basename=wfdb_basename,
            plot_folder=plot_folder,
            file_id=file_id
        )
    # Upload plot to 'plots/<file_id>/'
        plot_filename = os.path.basename(plot_path)
        supabase_plot_path = f"{PLOTS_FOLDER}/{file_id}/{plot_filename}"
        upload_file_to_supabase(bucket=SUPABASE_BUCKET, 
                               local_path=plot_path, 
                               remote_path=supabase_plot_path)

        plot_url = generate_signed_url_from_supabase(supabase_plot_path)

    except Exception as e:
        logger.exception('WFDB analysis failed')
        return jsonify({'error': f'Analysis failed: {e}'}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if plot_folder:
            shutil.rmtree(plot_folder, ignore_errors=True)

    return jsonify({'summary': summary, 'plot': plot_url}), 200

@wfdb_bp.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(current_app.config['PLOTS_DIR'], filename)

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