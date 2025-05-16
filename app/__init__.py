from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
import os

from .config import Config
from .models import db
from .routes.auth import auth_bp
from .routes.wfdb import wfdb_bp

# Initialize extensions
jwt = JWTManager()
migrate = Migrate()

# Ensure upload, wfdb, and plots directories exist
def _ensure_dirs(app):
    for folder in (
        app.config['UPLOAD_DIR'],
        app.config['WFDB_DIR'],
        app.config['PLOTS_DIR']
    ):
        os.makedirs(folder, exist_ok=True)

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

    # Create database tables
    with app.app_context():
        db.create_all()

    return app