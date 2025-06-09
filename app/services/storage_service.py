import os
import logging
from flask import current_app
from storage3.exceptions import StorageApiError

class StorageService:
    @staticmethod
    def get_supabase():
        supabase = getattr(current_app, "supabase", None)
        if supabase is None:
            raise RuntimeError("Supabase initialization failed.")
        return supabase

    @staticmethod
    def upload_file_to_supabase(
        local_path: str, storage_path: str, bucket_name="ecg-data"
    ) -> str:
        """
        Uploads a file to supabase bucket
        """
        supabase = StorageService.get_supabase()
        with open(local_path, "rb") as f:
            res = supabase.storage.from_(bucket_name).upload(
                path=storage_path, file=f, file_options={"upsert": "true"}
            )

        # Verify upload success
        if hasattr(res, "path") and res.path:
            return res.path
        else:
            raise Exception(f"Upload failed, response: {res}")

    @staticmethod
    def generate_signed_url(storage_path: str, bucket_name="ecg-data", expires_in=3600) -> str:
        """
        Generates a signed url to add data to supabase bucket
        """
        storage_path = storage_path.lstrip("/")
        if storage_path.startswith("app/"):
            storage_path = storage_path[4:]
        try:
            supabase = StorageService.get_supabase()
            res = supabase.storage.from_(bucket_name).create_signed_url(storage_path, expires_in)
            return res["signedURL"]
        except StorageApiError as e:
            logging.warning(f"Could not generate signed URL for {storage_path}: {e}")
            return None

    # @staticmethod
    # def get_ecg_signed_urls(ecg) -> tuple[str, str]:
    #     """
    #     Creates a signed url for supabase to retrieve ecg files
    #     """
    #     supabase = StorageService.get_supabase()
    #     bucket = "ecg-data"
    #     return (
    #         supabase.storage.from_(bucket).create_signed_url(ecg.wfdb_path, 3600)["signedURL"],
    #         supabase.storage.from_(bucket).create_signed_url(ecg.plot_path, 3600)["signedURL"],
    #     )
