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

    # --- Overall Status Determination ---
    if db_status == "healthy" and fs_status == "healthy":
        overall_status = "healthy"
        response_code = 200
        logger.info("Overall health status: HEALTHY.")
    else:
        overall_status = "unhealthy"
        response_code = 503
        logger.warning(f"Overall health status: UNHEALTHY. Database: {db_status}")

    return (
        jsonify(
            {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0", 
                "services": {"database": db_status},
                "uptime": "running",
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
    logger.debug("Application process is alive and responsive.")
    return jsonify({"status": "alive"}), 200