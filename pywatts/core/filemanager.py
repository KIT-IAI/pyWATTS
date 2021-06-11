import glob
import logging
import os
from datetime import datetime

from pywatts.core.exceptions.io_exceptions import IOException

logger = logging.getLogger()

ALLOWED_FILES = ["png", "csv", "xlsx", "pickle", "tex", "json", "h5", "pt", "md"]


class FileManager:
    """
    This class is responsible for managing files in pyWATTS.
    It ensures that all files for one pipeline run are in the same folder.
    Moreover, it appends a timestamp to the corresponding path

    :param path: Root path for the results of the pipeline
    :type path: str
    :param time_mode: If true, then a subfolder with the current time is created
    :type time_mode: bool
    """

    def __init__(self, path, time_mode=True):
        self.basic_path = path
        self.time_mode = time_mode
        if time_mode:
            self.path = os.path.join(path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        else:
            self.path = path

    def _create_path_dirs(self):
        """
        Creates all directories needed to write files to the directory at self.path
        """
        if self.time_mode:
            os.makedirs(self.path, exist_ok=False)
        else:
            os.makedirs(self.path, exist_ok=True)
        logger.info("Created folder %s", self.path)

    def get_path(self, filename: str, path=None):
        """
        Returns a path to file. This path is in the folder of the corresponding pipeline run.
        Moreover it is ensured that no data are overwritten.

        :param filename: Name of the file to write
        :type filename: str
        :param path: Optional path extension to the file.
        :return: The path, where the results should be stored.
        """
        if not os.path.exists(self.path):
            self._create_path_dirs()

        if filename.split(".")[-1] not in ALLOWED_FILES:
            message = f"{filename.split('.')[-1]} is not an allowed file type. Allowed types are {ALLOWED_FILES}."
            logger.error(message)
            raise IOException(message)
        if os.path.split(filename)[0] != "":
            logger.warning("Remove head of %s, since this contains path informations.", filename)
            filename = os.path.split(filename)[1]
        if path is not None:
            path = os.path.join(self.path, path)
            os.makedirs(path)
            logger.info("Created folder %s", path)
        else:
            path = self.path
        return_path = os.path.join(path, filename)
        if os.path.isfile(return_path):
            filename, extension = os.path.splitext(return_path)
            number = len(glob.glob(f'{filename}*{extension}'))
            logger.info("File %s already exists. We appended %s to the name", return_path, number + 1)
            return_path = f"{filename}_{number + 1}.{extension}"
        return return_path
