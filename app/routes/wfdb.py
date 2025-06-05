from flask import Blueprint, request, jsonify, url_for, send_from_directory, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, uuid, zipfile, shutil, logging
from datetime import datetime

from ..utils import analyze_and_plot, load_and_clean_all_leads
from ..models import db, ECG, User

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
    
    file_id = str(uuid.uuid4())
    zip_path = os.path.join(current_app.config['UPLOAD_DIR'], f"{file_id}.zip")
    f.save(zip_path)
    rec_dir = os.path.join(current_app.config['WFDB_DIR'], file_id)
    os.makedirs(rec_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(rec_dir)
    
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        shutil.rmtree(rec_dir, ignore_errors=True)
        os.remove(zip_path)
        return jsonify({'error': 'No .hea found'}), 400
    
    wfdb_basename = os.path.join(rec_dir, os.path.splitext(hea)[0])
    try:
        summary, plot_path = analyze_and_plot(
            wfdb_basename=wfdb_basename,
            plot_folder=current_app.config['PLOTS_DIR'],
            file_id=file_id
        )
        
        # Get form data with proper type conversion and null handling
        patient_name = request.form.get('patient_name')
        age = request.form.get('age')
        age = int(age) if age and age.isdigit() else None
        gender = request.form.get('gender')
        
        # Create ECG record in database with null handling
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
            wfdb_path=rec_dir,
            plot_path=plot_path
        )
        
        db.session.add(ecg)
        db.session.commit()
        
        plot_url = url_for('wfdb.serve_plot', filename=os.path.basename(plot_path), _external=True)
        return jsonify({
            'file_id': file_id,
            'summary': summary,
            'plot': plot_url,
            'record': ecg.to_dict()
        }), 201
        
    except Exception as e:
        logger.exception('WFDB analysis failed on upload')
        return jsonify({'error': f'Analysis failed: {e}'}), 500

@wfdb_bp.route('/record/<file_id>', methods=['GET'])
@jwt_required()
def get_record(file_id):
    user_id = int(get_jwt_identity())
    ecg = ECG.query.filter_by(file_id=file_id, user_id=user_id).first()
    
    if not ecg:
        return jsonify({'error': 'Record not found'}), 404
    
    # Get plot URL
    plot_url = url_for('wfdb.serve_plot', filename=os.path.basename(ecg.plot_path), _external=True)
    
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
    rec_dir = os.path.join(current_app.config['WFDB_DIR'], file_id)
    if not os.path.isdir(rec_dir):
        return jsonify({'error': 'Record not found'}), 404
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        return jsonify({'error': 'No .hea found'}), 400
    wfdb_basename = os.path.join(rec_dir, os.path.splitext(hea)[0])
    try:
        summary, plot_path = analyze_and_plot(
            wfdb_basename=wfdb_basename,
            plot_folder=current_app.config['PLOTS_DIR'],
            file_id=file_id
        )
    except Exception as e:
        logger.exception('WFDB analysis failed')
        return jsonify({'error': f'Analysis failed: {e}'}), 500
    plot_url = url_for('wfdb.serve_plot', filename=os.path.basename(plot_path), _external=True)
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
        data = load_and_clean_all_leads(ecg.wfdb_path)
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

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.exception(f"Failed to load cleaned ECG leads for {file_id}")
        return jsonify({'error': f"Failed to load ECG leads: {e}"}), 500