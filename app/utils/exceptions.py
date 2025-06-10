class APIError(Exception):
    def __init__(self, code, message, status_code=400, details=None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

class ErrorCodes:
    # Validation
    MISSING_FIELD = "validation_missing_field"
    INVALID_FORMAT = "validation_invalid_format" 
    FILE_TOO_LARGE = "validation_file_too_large"
    INVALID_FILE_TYPE = "validation_invalid_file_type"
    
    # ECG Processing
    ECG_CORRUPTED = "ecg_file_corrupted"
    ECG_PROCESSING_FAILED = "ecg_processing_failed"
    ECG_INSUFFICIENT_DATA = "ecg_insufficient_data"
    
    # Database
    RECORD_NOT_FOUND = "record_not_found"
    DUPLICATE_ENTRY = "duplicate_entry"
    
    # System
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
