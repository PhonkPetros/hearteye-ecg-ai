from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from sqlalchemy import text
from supabase import create_client, Client
import os
import logging
import sys
from pythonjsonlogger import jsonlogger
from .config import Config
from .models import db
from .routes.auth import auth_bp
from .routes.ecg import ecg_bp
from .routes.health import health_bp
from .services.prediction_service import PredictionService
from .utils.error_handlers import register_error_handlers
from .logging_config import setup_logging

# Initialize extensions
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # Setup logging
    setup_logging(app.import_name) 

    # Load config
    config = {key: getattr(Config, key) for key in dir(Config) if key.isupper()}
    app.config.update(config)

    # Initialize extensions
    CORS(app, supports_credentials=True, origins=app.config.get("CORS_ORIGINS", []))
    jwt.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)

    # Initialize Supabase
    app.supabase = create_client(app.config["SUPABASE_URL"], app.config["SUPABASE_KEY"])

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(ecg_bp)
    app.register_blueprint(health_bp)

    # Load ML model
    model_path = "models/rf_rr_hr_optimized_model.pkl"
    PredictionService.load_model(model_path)
    
    # Register error handlers
    register_error_handlers(app)

    return app