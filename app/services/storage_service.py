import os
import logging
from flask import current_app
from storage3.exceptions import StorageApiError
from ..utils.exceptions import APIError, ErrorCodes

logger = logging.getLogger(__name__)

class StorageService:
    @staticmethod
    def get_supabase():
        supabase = getattr(current_app, "supabase", None)
        if supabase is None:
            logger.error("Supabase client not initialized in Flask app context.")
            raise APIError(
                ErrorCodes.SERVICE_UNAVAILABLE,
                "Supabase service is not properly configured.",
                status_code=500
            )
        return supabase

    @staticmethod
    def upload_file_to_supabase(
        local_path: str, storage_path: str, bucket_name="ecg-data"
    ) -> str:
        """
        Uploads a file to supabase bucket
        """
        try:
            supabase = StorageService.get_supabase()
            with open(local_path, "rb") as f:
                logger.info(f"Attempting to upload file from {local_path} to Supabase bucket '{bucket_name}' at {storage_path}")
                res = supabase.storage.from_(bucket_name).upload(
                    path=storage_path, file=f, file_options={"upsert": "true"}
                )

            # Verify upload success
            if hasattr(res, "path") and res.path:
                logger.info(f"File uploaded successfully to Supabase: {res.path}", extra={"storage_path": res.path, "bucket": bucket_name})
                return res.path
            else:
                # This could be a malformed response or an unknown error
                logger.error(f"Supabase upload returned an unexpected response: {res}", extra={"local_path": local_path, "storage_path": storage_path})
                raise APIError(
                    ErrorCodes.INTERNAL_ERROR,
                    "Failed to upload file to storage: Unexpected response.",
                    status_code=500,
                    details={"response": str(res)}
                )
        except FileNotFoundError:
            logger.warning(f"Attempted to upload non-existent file: {local_path}")
            raise APIError(
                ErrorCodes.RECORD_NOT_FOUND,
                f"Source file not found for upload: {local_path}",
                status_code=404,
                details={"file_path": local_path}
            )
        except StorageApiError as e:
            # Catch Supabase specific errors and convert to APIError
            logger.exception(f"Supabase Storage API error during upload for {storage_path}", exc_info=e)
            raise APIError(
                ErrorCodes.SERVICE_UNAVAILABLE, # Or a more specific code if Supabase errors can be parsed
                f"Supabase storage error: {e.message if hasattr(e, 'message') else str(e)}",
                status_code=500,
                details={"error_type": type(e).__name__, "message": str(e)}
            )
        except Exception as e:
            # Catch any other unexpected errors during the process
            logger.exception(f"An unexpected error occurred during file upload for {local_path}", exc_info=e)
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                "An unexpected error occurred during file upload.",
                status_code=500,
                details={"error_type": type(e).__name__, "message": str(e)}
            )

    @staticmethod
    def generate_signed_url(storage_path: str, bucket_name="ecg-data", expires_in=3600) -> str:
        """
        Generates a signed url to access data from a Supabase bucket
        """
        original_storage_path = storage_path
        storage_path = storage_path.lstrip("/")
        if storage_path.startswith("app/"):
            storage_path = storage_path[4:] # Remove 'app/' prefix if present

        try:
            supabase = StorageService.get_supabase()
            logger.info(f"Attempting to generate signed URL for '{storage_path}' in bucket '{bucket_name}'")
            res = supabase.storage.from_(bucket_name).create_signed_url(storage_path, expires_in)
            
            if "signedURL" in res and res["signedURL"]:
                logger.info(f"Successfully generated signed URL for {storage_path}", extra={"url_path": storage_path, "bucket": bucket_name})
                return res["signedURL"]
            else:
                logger.warning(f"Supabase returned no signed URL: {res}", extra={"storage_path": original_storage_path, "response": res})
                raise APIError(
                    ErrorCodes.INTERNAL_ERROR,
                    f"Failed to generate signed URL for {original_storage_path}: No URL in response.",
                    status_code=500,
                    details={"storage_path": original_storage_path, "supabase_response": res}
                )

        except StorageApiError as e:
            logger.warning(f"Supabase Storage API error generating signed URL for {original_storage_path}: {e}", exc_info=e)
            raise APIError(
                ErrorCodes.SERVICE_UNAVAILABLE,
                f"Failed to generate signed URL due to storage error: {e.message if hasattr(e, 'message') else str(e)}",
                status_code=500,
                details={"storage_path": original_storage_path, "error_type": type(e).__name__, "message": str(e)}
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred while generating signed URL for {original_storage_path}", exc_info=e)
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                "An unexpected error occurred during signed URL generation.",
                status_code=500,
                details={"storage_path": original_storage_path, "error_type": type(e).__name__, "message": str(e)}
            )