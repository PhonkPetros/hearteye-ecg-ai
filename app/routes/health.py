from flask import Blueprint, jsonify
from datetime import datetime
from sqlalchemy import text
from ..models import db
import os

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and load balancing"""
    try:
        # Check database connection
        with db.engine.connect() as connection:
            connection.execute(text('SELECT 1'))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check file system
    upload_dir = os.environ.get('UPLOAD_DIR', '/app/uploads')
    wfdb_dir = os.environ.get('WFDB_DIR', '/app/wfdb')
    plots_dir = os.environ.get('PLOTS_DIR', '/app/plots')
    
    fs_status = "healthy"
    if not all(os.path.exists(d) for d in [upload_dir, wfdb_dir, plots_dir]):
        fs_status = "unhealthy: missing directories"
    
    # Overall health
    overall_status = "healthy" if all("healthy" in status for status in [db_status, fs_status]) else "unhealthy"
    
    return jsonify({
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": db_status,
            "filesystem": fs_status
        },
        "uptime": "running"
    }), 200 if overall_status == "healthy" else 503

@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check for Kubernetes/container orchestration"""
    try:
        # Check if application is ready to serve requests
        with db.engine.connect() as connection:
            connection.execute(text('SELECT 1'))
        return jsonify({"status": "ready"}), 200
    except Exception:
        return jsonify({"status": "not ready"}), 503

@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """Liveness check for Kubernetes/container orchestration"""
    # Simple check that the application is running
    return jsonify({"status": "alive"}), 200 