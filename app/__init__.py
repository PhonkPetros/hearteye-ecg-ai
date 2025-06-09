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

# Initialize extensions
jwt = JWTManager()
migrate = Migrate()

# Application factory
def create_app():
    app = Flask(__name__)

    logger = logging.getLogger()
    if not logger.hasHandlers():
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(lineno)d'
        )
        logHandler.setFormatter(formatter)
        logger.addHandler(logHandler)
        logger.setLevel(logging.INFO)

    # Load config from Config class directly
    config = {key: getattr(Config, key) for key in dir(Config) if key.isupper()}
    app.config.update(config)

    CORS(app, supports_credentials=True, origins=app.config.get("CORS_ORIGINS", []))
    jwt.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)

    app.supabase = create_client(app.config["SUPABASE_URL"], app.config["SUPABASE_KEY"])

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(ecg_bp)
    app.register_blueprint(health_bp)

    return app
