from flask import Flask, request, jsonify
import os, uuid, zipfile, shutil, logging
import ecg_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

app = Flask(__name__)

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
WFDB_DIR   = os.path.join(BASE_DIR, "wfdb_records")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
for d in (UPLOAD_DIR, WFDB_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_wfdb():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.zip'):
        return jsonify({"error": "ZIP required"}), 400

    file_id  = str(uuid.uuid4())
    zip_path = os.path.join(UPLOAD_DIR, f"{file_id}.zip")
    f.save(zip_path)

    rec_dir = os.path.join(WFDB_DIR, file_id)
    os.makedirs(rec_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(rec_dir)

    # sanity check
    hea = next((fn for fn in os.listdir(rec_dir) if fn.lower().endswith('.hea')), None)
    if not hea:
        shutil.rmtree(rec_dir, ignore_errors=True)
        os.remove(zip_path)
        return jsonify({"error": "No .hea found"}), 400

    return jsonify({"file_id": file_id}), 201


@app.route('/analyze_wfdb/<file_id>', methods=['GET'])
def analyze_wfdb(file_id):
    rec_dir = os.path.join(WFDB_DIR, file_id)
    if not os.path.isdir(rec_dir):
        return jsonify({"error":"Record not found"}), 404

    hea = next((f for f in os.listdir(rec_dir) if f.lower().endswith('.hea')), None)
    wfdb_path = os.path.join(rec_dir, os.path.splitext(hea)[0])

    try:
        summary = ecg_utils.analyze_wfdb_and_plot_summary(
            wfdb_basename=wfdb_path,
            plot_folder=PLOTS_DIR,
            file_id=file_id
        )
    except Exception as e:
        logging.exception("WFDB analysis failed")
        return jsonify({"error":f"Analysis failed: {e}"}), 500

    return jsonify({"summary": summary}), 200


if __name__ == '__main__':
    app.run(debug=True)
