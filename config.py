class Config:
    JWT_SECRET_KEY = "dev-secret-key"
    JWT_TOKEN_LOCATION = ['cookies']
    JWT_COOKIE_SECURE = "True" # Set to True in production
    JWT_COOKIE_HTTPONLY = True
    JWT_ACCESS_COOKIE_PATH = '/'
    JWT_COOKIE_SAMESITE = "Lax"

    # Add other Flask config if needed
    DEBUG = "True"
    CORS_SUPPORTS_CREDENTIALS = True
