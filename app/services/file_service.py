import wfdb
import numpy as np
import pyedflib
import os
import re
import logging
from ..utils.exceptions import APIError, ErrorCodes
from .ecg_processing_service import ECGProcessingService

logger = logging.getLogger(__name__)

class FileHandlingService:
    @staticmethod
    def load_and_clean_all_leads(wfdb_path: str) -> dict:
        """
        Load and clean all leads from a WFDB record.
        Args:
            wfdb_path (str): The path to the WFDB record directory or .hea file (without extension).
        Returns:
            dict: A dictionary containing 'cleaned_signals' (numpy array), 'fs' (sampling frequency),
                  and 'lead_names' (list of strings).
        Raises:
            APIError: If the WFDB file/directory is not found, is corrupted, or any processing error occurs.
        """
        logger.info(f"Attempting to load and clean WFDB record from: {wfdb_path}")
        wfdb_basename = None

        try:
            if not os.path.isdir(wfdb_path):
                # If path is to a .hea file or a base name without extension
                wfdb_basename = os.path.splitext(wfdb_path)[0]
                if not os.path.exists(f"{wfdb_basename}.hea"):
                    logger.warning(f"WFDB .hea file not found at expected path: {wfdb_basename}.hea")
                    raise FileNotFoundError(f"WFDB .hea file not found: {wfdb_basename}.hea")
            else:
                # If path is a directory, find the .hea file within it
                hea_file = next(
                    (f for f in os.listdir(wfdb_path) if f.lower().endswith(".hea")), None
                )
                if not hea_file:
                    logger.warning(f"No .hea file found in directory: {wfdb_path}")
                    raise FileNotFoundError(f"No .hea file found in directory {wfdb_path}")
                wfdb_basename = os.path.join(wfdb_path, os.path.splitext(hea_file)[0])

            # Call ECGProcessingService to read the record
            signals, fs, record = ECGProcessingService.read_wfdb_record(wfdb_basename)
            
            lead_names = (
                record.sig_name
                if record.sig_name
                else [f"Lead_{i}" for i in range(signals.shape[1])]
            )

            logger.info(f"Successfully loaded and cleaned WFDB record from {wfdb_path}",
                        extra={"sampling_frequency": fs, "num_leads": signals.shape[1], "lead_names": lead_names})
            return {"cleaned_signals": signals, "fs": fs, "lead_names": lead_names}

        except FileNotFoundError as e:
            logger.warning(f"File not found error while loading WFDB record: {wfdb_path}. Error: {e}", exc_info=e)
            raise APIError(
                ErrorCodes.RECORD_NOT_FOUND,
                f"WFDB record or .hea file not found at {wfdb_path}.",
                status_code=404,
                details={"path": wfdb_path, "error_detail": str(e)}
            )
        except wfdb.io.annotation.AnnotationError as e:
            logger.error(f"WFDB annotation error for {wfdb_path}: {e}", exc_info=e)
            raise APIError(
                ErrorCodes.ECG_CORRUPTED,
                f"WFDB record at {wfdb_path} is corrupted or has an invalid annotation format.",
                status_code=422, # Unprocessable Entity
                details={"path": wfdb_path, "error_detail": str(e)}
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading/cleaning WFDB record from {wfdb_path}", exc_info=e)
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                f"Failed to load and clean WFDB record: An unexpected internal error occurred.",
                status_code=500,
                details={"path": wfdb_path, "error_type": type(e).__name__, "message": str(e)}
            )

    @staticmethod
    def convert_edf_to_wfdb(edf_path: str, output_dir: str = None) -> str:
        """
        Converts an EDF file to WFDB format.
        Args:
            edf_path (str): The path to the input EDF file.
            output_dir (str, optional): The directory where the WFDB files will be saved.
                                        Defaults to the same directory as the EDF file.
        Returns:
            str: The full path to the generated WFDB record (basename, e.g., '/path/to/record_name').
        Raises:
            APIError: If the EDF file is not found, is corrupted, or conversion fails.
        """
        logger.info(f"Attempting to convert EDF file '{edf_path}' to WFDB format.")

        def sanitize_filename(name: str) -> str:
            sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
            sanitized = sanitized.strip("_")
            return sanitized or "default_record_name"

        try:
            if not os.path.exists(edf_path):
                logger.warning(f"EDF file not found: {edf_path}")
                raise FileNotFoundError(f"EDF file not found: {edf_path}")

            if output_dir is None:
                output_dir = os.path.dirname(edf_path)
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Ensuring output directory for WFDB is: {output_dir}")

            filename_base = os.path.splitext(os.path.basename(edf_path))[0]
            record_name = sanitize_filename(filename_base)
            full_output_wfdb_path = os.path.join(output_dir, record_name)

            edf_reader = pyedflib.EdfReader(edf_path)
            num_signals = edf_reader.signals_in_file
            fs = edf_reader.getSampleFrequency(0)
            signal_labels = edf_reader.getSignalLabels()
            
            # Read signals and ensure they are all loaded before closing
            signals = [edf_reader.readSignal(i) for i in range(num_signals)]
            edf_reader.close()
            del edf_reader 

            # Trim signals to shortest length to avoid issues with wfdb.wrsamp
            min_len = min(len(s) for s in signals)
            signals_trimmed = [s[:min_len] for s in signals]
            logger.info(f"EDF file read successfully. Signals trimmed to length: {min_len}")

            # Save as WFDB
            wfdb.wrsamp(
                record_name=record_name,
                fs=fs,
                units=["uV"] * num_signals, 
                sig_name=signal_labels,
                p_signal=np.column_stack(signals_trimmed),
                write_dir=output_dir,
            )

            logger.info(f"Successfully converted EDF '{edf_path}' to WFDB at '{full_output_wfdb_path}'",
                        extra={"edf_path": edf_path, "wfdb_path": full_output_wfdb_path})
            return full_output_wfdb_path

        except FileNotFoundError as e:
            logger.warning(f"Input EDF file not found: {edf_path}", exc_info=e)
            raise APIError(
                ErrorCodes.RECORD_NOT_FOUND,
                f"Input EDF file not found: {edf_path}.",
                status_code=404,
                details={"file_path": edf_path, "error_detail": str(e)}
            )
        except pyedflib.EdfException as e:
            logger.error(f"EDF parsing error for {edf_path}: {e}", exc_info=e)
            raise APIError(
                ErrorCodes.ECG_CORRUPTED,
                f"EDF file at {edf_path} is corrupted or has an invalid format.",
                status_code=422, # Unprocessable Entity
                details={"file_path": edf_path, "error_detail": str(e)}
            )
        except Exception as e:
            # Catch any other unexpected errors during the conversion process
            logger.exception(f"An unexpected error occurred during EDF to WFDB conversion for {edf_path}", exc_info=e)
            raise APIError(
                ErrorCodes.INTERNAL_ERROR,
                "An unexpected error occurred during file conversion.",
                status_code=500,
                details={"file_path": edf_path, "error_type": type(e).__name__, "message": str(e)}
            )