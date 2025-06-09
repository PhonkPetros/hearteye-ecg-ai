# config.py
import os
from datetime import timedelta


class Config:
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "postgresql://hearteye:hearteye@localhost:5432/hearteye"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-please-change")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-secret-key-please-change")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
