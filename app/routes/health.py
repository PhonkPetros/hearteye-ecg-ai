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
    logger.info("Health check requested")

    try:
        # Check database connection
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        db_status = "healthy"
        logger.info("Database health check passed")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        logger.error("Database health check failed", exc_info=e)

    upload_dir = os.environ.get("UPLOAD_DIR", "/app/uploads")
    wfdb_dir = os.environ.get("WFDB_DIR", "/app/wfdb")
    plots_dir = os.environ.get("PLOTS_DIR", "/app/plots")

    fs_status = "healthy"
    missing_dirs = [d for d in [upload_dir, wfdb_dir, plots_dir] if not os.path.exists(d)]
    if missing_dirs:
        fs_status = f"unhealthy: missing directories {missing_dirs}"
        logger.error(f"Filesystem health check failed, missing dirs: {missing_dirs}")
    else:
        logger.info("Filesystem health check passed")

    overall_status = (
        "healthy"
        if all("healthy" in status for status in [db_status, fs_status])
        else "unhealthy"
    )
    logger.info(f"Overall health status: {overall_status}")

    return (
        jsonify(
            {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "services": {"database": db_status, "filesystem": fs_status},
                "uptime": "running",
            }
        ),
        200 if overall_status == "healthy" else 503,
    )


@health_bp.route("/ready", methods=["GET"])
def readiness_check():
    """Readiness check for Kubernetes/container orchestration"""
    logger.info("Readiness check requested")
    try:
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Readiness check passed")
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        logger.error("Readiness check failed", exc_info=e)
        return jsonify({"status": "not ready"}), 503


@health_bp.route("/live", methods=["GET"])
def liveness_check():
    """Liveness check for Kubernetes/container orchestration"""
    logger.info("Liveness check requested")
    # Simple check that the application is running
    return jsonify({"status": "alive"}), 200
