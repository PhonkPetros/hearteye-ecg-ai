import logging.config
import os
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(app_name):
    """
    Configures the application's logging using dictConfig.
    Args:
        app_name (str): The import name of the Flask application, used to target its logger.
    """

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False, 
        'formatters': {
            'json_formatter': {
                '()': jsonlogger.JsonFormatter,
                'format': '%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(lineno)d',
            },
            'standard_formatter': { # For non-JSON logs if needed (e.g. for gunicorn default)
                'format': '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'json_formatter',
                'stream': 'ext://sys.stdout', # Explicitly send to stdout
            },
        },
        'loggers': {
            app_name: { # This targets your Flask app's logger (e.g., 'app')
                'handlers': ['console'],
                'level': os.environ.get('LOG_LEVEL', 'INFO').upper(),
                'propagate': False, # **Crucial: prevent propagation to root logger**
            },
            'gunicorn.access': { # Gunicorn's access logger
                'handlers': ['console'], # You can choose 'standard_formatter' here if you want default Gunicorn format
                'level': 'INFO',
                'propagate': False, # Prevent Gunicorn access logs from polluting root
            },
            'gunicorn.error': { # Gunicorn's error logger
                'handlers': ['console'],
                'level': os.environ.get('LOG_LEVEL', 'INFO').upper(),
                'propagate': False, # Prevent Gunicorn error logs from polluting root
            },
            'werkzeug': { # Flask's development server logger
                'handlers': ['console'],
                'level': 'WARNING', # Keep this less verbose
                'propagate': False,
            },
            'sqlalchemy.engine': { # SQLAlchemy query logs
                'handlers': ['console'],
                'level': 'WARNING', # Suppress verbose SQL logs by default
                'propagate': False,
            },
            # Add other noisy third-party loggers here as needed
            'httpx': {
                'handlers': ['console'],
                'level': 'INFO', # Keep info logs for HTTP requests
                'propagate': False,
            }
        },
        'root': { # The root logger - set a default, but most logs should go via specific loggers
            'handlers': ['console'], 
            'level': os.environ.get('LOG_LEVEL', 'INFO').upper(),
            'propagate': False,
        }
    }
    
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
        # Optional: You can get a reference to the app's logger here for immediate use
        # app_logger = logging.getLogger(app_name)
        # app_logger.info("Logging configured successfully.")
    except Exception as e:
        # Fallback if logging config itself fails
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logging.error(f"Failed to load logging configuration: {e}", exc_info=True)
