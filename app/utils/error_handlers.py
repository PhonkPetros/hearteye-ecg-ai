from flask import jsonify, request
from marshmallow import ValidationError as MarshmallowValidationError
from werkzeug.exceptions import HTTPException
from functools import wraps
import logging
from .exceptions import APIError, ErrorCodes

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(e):
        logger.warning(f"API Error: {e.code} - {e.message}", extra={
            "error_code": e.code,
            "details": e.details,
            "status_code": e.status_code,
            "path": request.path,
            "method": request.method
        })

        response = {
            "error": {
                "code": e.code,
                "message": e.message
            }
        }

        if e.details:
            response["error"]["details"] = e.details

        return jsonify(response), e.status_code

    @app.errorhandler(MarshmallowValidationError)
    def handle_validation_error(e):
        logger.warning("Marshmallow validation error caught by global handler", extra={
            "messages": e.messages,
            "path": request.path,
            "method": request.method
        })
        return jsonify({
            "error": {
                "code": ErrorCodes.INVALID_FORMAT,
                "message": "Validation failed",
                "details": e.messages
            }
        }), 400

    @app.errorhandler(HTTPException)
    def handle_http_error(e):
        logger.warning(f"HTTP error: {e.code}", extra={
            "description": e.description,
            "path": request.path,
            "method": request.method
        })
        return jsonify({
            "error": {
                "code": f"http_{e.code}",
                "message": e.description
            }
        }), e.code

    @app.errorhandler(Exception)
    def handle_generic_error(e):
        logger.exception("Unhandled exception caught by global handler", exc_info=e, extra={
            "path": request.path,
            "method": request.method
        })
        return jsonify({
            "error": {
                "code": ErrorCodes.INTERNAL_ERROR,
                "message": "An unexpected error occurred"
            }
        }), 500

def handle_errors(func):
    """Decorator for common error handling in routes"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError:
            raise 
        except MarshmallowValidationError as e:
            logger.warning("Marshmallow validation error caught by decorator", extra={"messages": e.messages})
            raise APIError(
                ErrorCodes.INVALID_FORMAT,
                "Validation failed",
                status_code=400,
                details=e.messages
            )
        except FileNotFoundError:
            raise APIError(
                ErrorCodes.RECORD_NOT_FOUND,
                "Requested resource not found",
                404
            )
        except ValueError as e:
            # Re-evaluate if this specific ValueError is still needed.
            # If so, keep it. Otherwise, it might be removed or refined.
            raise APIError(
                ErrorCodes.INVALID_FORMAT,
                f"Invalid input: {str(e)}",
                400
            )
        except Exception as e:
            logger.exception("Unexpected error in route caught by decorator", exc_info=e)
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                "An unexpected error occurred",
                500
            )
    return wrapper