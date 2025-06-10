import logging
from flask import Blueprint, jsonify
from datetime import datetime
from sqlalchemy import text
from ..models import db
import os

health_bp = Blueprint("health", __name__)
logger = logging.getLogger(__name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring and load balancing"""
    logger.info("Health check requested.")

    db_status = "unknown"
    fs_status = "unknown"
    overall_status = "unhealthy" # Default to unhealthy until proven otherwise
    response_code = 503 # Default HTTP status code to Service Unavailable

    # --- Database Health Check ---
    try:
        logger.debug("Attempting database connection check for health endpoint.")
        # Attempt to connect and execute a simple query
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        db_status = "healthy"
        logger.info("Database health check passed.")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}", exc_info=True) # Log full traceback

    # --- Filesystem Health Check ---
    upload_dir = os.environ.get("UPLOAD_DIR", "/app/uploads")
    wfdb_dir = os.environ.get("WFDB_DIR", "/app/wfdb")
    plots_dir = os.environ.get("PLOTS_DIR", "/app/plots")
    
    logger.debug(f"Checking filesystem directories: UPLOAD_DIR='{upload_dir}', WFDB_DIR='{wfdb_dir}', PLOTS_DIR='{plots_dir}'.")

    missing_dirs = []
    # Check for existence of each directory
    if not os.path.exists(upload_dir):
        missing_dirs.append(upload_dir)
        logger.warning(f"Upload directory '{upload_dir}' is missing.")
    if not os.path.exists(wfdb_dir):
        missing_dirs.append(wfdb_dir)
        logger.warning(f"WFDB directory '{wfdb_dir}' is missing.")
    if not os.path.exists(plots_dir):
        missing_dirs.append(plots_dir)
        logger.warning(f"Plots directory '{plots_dir}' is missing.")

    if missing_dirs:
        fs_status = f"unhealthy: missing directories {', '.join(missing_dirs)}"
        logger.error(f"Filesystem health check failed, missing directories: {missing_dirs}.")
    else:
        fs_status = "healthy"
        logger.info("Filesystem health check passed.")

    # --- Overall Status Determination ---
    if db_status == "healthy" and fs_status == "healthy":
        overall_status = "healthy"
        response_code = 200
        logger.info("Overall health status: HEALTHY.")
    else:
        overall_status = "unhealthy"
        response_code = 503
        logger.warning(f"Overall health status: UNHEALTHY. Database: {db_status}, Filesystem: {fs_status}.")

    return (
        jsonify(
            {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0", # Assuming a fixed version for this example
                "services": {"database": db_status, "filesystem": fs_status},
                "uptime": "running", # Placeholder; actual uptime calculation might be more complex
            }
        ),
        response_code,
    )

@health_bp.route("/ready", methods=["GET"])
def readiness_check():
    """Readiness check for Kubernetes/container orchestration"""
    logger.info("Readiness check requested.")
    try:
        logger.debug("Attempting database connection for readiness check.")
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Readiness check passed: Database is accessible.")
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        logger.error(f"Readiness check failed: Database not accessible. Error: {e}", exc_info=True)
        return jsonify({"status": "not ready", "reason": f"Database not accessible: {str(e)}"}), 503

@health_bp.route("/live", methods=["GET"])
def liveness_check():
    """Liveness check for Kubernetes/container orchestration"""
    logger.info("Liveness check requested.")
    # For a liveness check, we typically just verify the application process is running and responsive.
    # No complex external dependencies checks are usually performed here.
    logger.debug("Application process is alive and responsive.")
    return jsonify({"status": "alive"}), 200