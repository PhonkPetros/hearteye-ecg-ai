from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from sqlalchemy import text
import os
import logging

from .config import Config
from .models import db
from .routes.auth import auth_bp
from .routes.wfdb import wfdb_bp
from .routes.health import health_bp

# Initialize extensions
jwt = JWTManager()
migrate = Migrate()

logger = logging.getLogger(__name__)

# Ensure upload, wfdb, and plots directories exist
def _ensure_dirs(app):
    for folder in (
        app.config['UPLOAD_DIR'],
        app.config['WFDB_DIR'],
        app.config['PLOTS_DIR']
    ):
        os.makedirs(folder, exist_ok=True)

# Auto-update database schema
def _update_database_schema():
    """Add missing columns to existing tables if they don't exist"""
    try:
        # Check if gender column exists in ecgs table
        with db.engine.connect() as connection:
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ecgs' AND column_name = 'gender'
            """))
            
            if not result.fetchone():
                logger.info("Adding missing 'gender' column to ecgs table")
                connection.execute(text("ALTER TABLE ecgs ADD COLUMN gender VARCHAR(1)"))
                connection.commit()
                logger.info("Successfully added 'gender' column to ecgs table")
            else:
                logger.info("Gender column already exists in ecgs table")
                
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")

# Application factory
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    CORS(app, supports_credentials=True, origins=app.config.get('CORS_ORIGINS', []))
    jwt.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)

    _ensure_dirs(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(wfdb_bp)
    app.register_blueprint(health_bp)

    # Create database tables and update schema
    with app.app_context():
        db.create_all()
        _update_database_schema()

    return app