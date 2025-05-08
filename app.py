from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from datetime import datetime
import os
import uuid
import zipfile
import shutil
import logging
import ecg_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

app = Flask(__name__)
CORS(app)

# Directories
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
WFDB_DIR   = os.path.join(BASE_DIR, "wfdb_records")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
for d in (UPLOAD_DIR, WFDB_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_wfdb():
    # Validate file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.zip'):
        return jsonify({"error": "ZIP required"}), 400
    file_id = str(uuid.uuid4())
    zip_path = os.path.join(UPLOAD_DIR, f"{file_id}.zip")
    f.save(zip_path)
    rec_dir = os.path.join(WFDB_DIR, file_id)
    os.makedirs(rec_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(rec_dir)
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        shutil.rmtree(rec_dir, ignore_errors=True)
        os.remove(zip_path)
        return jsonify({"error": "No .hea found"}), 400
    wfdb_path = os.path.join(rec_dir, os.path.splitext(hea)[0])
    try:
        summary, plot_path = ecg_utils.analyze_wfdb_and_plot_summary(
            wfdb_basename=wfdb_path,
            plot_folder=PLOTS_DIR,
            file_id=file_id
        )
    except Exception as e:
        logging.exception("WFDB analysis failed on upload")
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    logging.info(f"Analysis summary for {file_id}: {summary}")
    plot_fname = os.path.basename(plot_path) if plot_path else None
    plot_url   = url_for('serve_plot', filename=plot_fname) if plot_fname else None
    return jsonify({
        "file_id": file_id,
        "summary": summary,
        "plot": plot_url
    }), 201

@app.route('/analyze_wfdb/<file_id>', methods=['GET'])
def analyze_wfdb(file_id):
    rec_dir = os.path.join(WFDB_DIR, file_id)
    if not os.path.isdir(rec_dir):
        return jsonify({"error": "Record not found"}), 404
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        return jsonify({"error": "No .hea found"}), 400
    wfdb_path = os.path.join(rec_dir, os.path.splitext(hea)[0])
    try:
        summary, plot_path = ecg_utils.analyze_wfdb_and_plot_summary(
            wfdb_basename=wfdb_path,
            plot_folder=PLOTS_DIR,
            file_id=file_id
        )
    except Exception as e:
        logging.exception("WFDB analysis failed")
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    logging.info(f"Analysis summary for {file_id}: {summary}")
    plot_fname = os.path.basename(plot_path) if plot_path else None
    plot_url   = url_for('serve_plot', filename=plot_fname) if plot_fname else None
    return jsonify({"summary": summary, "plot": plot_url}), 200

# Serve ECG waveform images
@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

# List and search past ECGs
@app.route('/history', methods=['GET'])
def history():
    search = request.args.get('search', '').lower()
    records = []
    for file_id in os.listdir(WFDB_DIR):
        rec_dir = os.path.join(WFDB_DIR, file_id)
        if not os.path.isdir(rec_dir):
            continue
        if search and search not in file_id.lower():
            continue
        hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
        if not hea:
            continue
        wfdb_path = os.path.join(rec_dir, os.path.splitext(hea)[0])
        summary, plot_path = ecg_utils.analyze_wfdb_and_plot_summary(
            wfdb_basename=wfdb_path,
            plot_folder=PLOTS_DIR,
            file_id=file_id
        )
        plot_fname = os.path.basename(plot_path) if plot_path else None
        plot_url   = url_for('serve_plot', filename=plot_fname)
        dt = datetime.fromtimestamp(os.path.getmtime(rec_dir)).isoformat()
        records.append({
            "id": file_id,
            "patientName": file_id,
            "date": dt,
            "classification": "",
            "summary": summary,
            "plot": plot_url
        })
    return jsonify(records), 200

if __name__ == '__main__':
    app.run(debug=True)
