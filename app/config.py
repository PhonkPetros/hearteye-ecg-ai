import os
from datetime import timedelta

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-please-change')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key-please-change')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'postgresql://hearteye:hearteye@localhost:5432/hearteye')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File storage configuration
    UPLOAD_DIR = os.environ.get('UPLOAD_DIR', 'uploads')
    WFDB_DIR = os.environ.get('WFDB_DIR', 'wfdb')
    PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots')
    
    # CORS configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
