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
            'standard_formatter': {
                'format': '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'json_formatter',
                'stream': 'ext://sys.stdout', 
            },
        },
        'loggers': {
            app_name: { 
                'handlers': ['console'],
                'level': os.environ.get('LOG_LEVEL', 'INFO').upper(),
                'propagate': False,
            },
            'gunicorn.access': { 
                'handlers': ['console'], 
                'level': 'INFO',
                'propagate': False,
            },
            'gunicorn.error': {
                'handlers': ['console'],
                'level': os.environ.get('LOG_LEVEL', 'INFO').upper(),
                'propagate': False, 
            },
            'werkzeug': {
                'handlers': ['console'],
                'level': 'WARNING', 
                'propagate': False,
            },
            'sqlalchemy.engine': { 
                'handlers': ['console'],
                'level': 'WARNING', 
                'propagate': False,
            },
            'httpx': {
                'handlers': ['console'],
                'level': 'INFO', 
                'propagate': False,
            }
        },
        'root': { 
            'handlers': ['console'], 
            'level': os.environ.get('LOG_LEVEL', 'INFO').upper(),
            'propagate': False,
        }
    }
    
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logging.error(f"Failed to load logging configuration: {e}", exc_info=True)
