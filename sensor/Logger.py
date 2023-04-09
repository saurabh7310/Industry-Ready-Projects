import logging
import os
from datetime import datetime

# CURRENT TIME FOR LOG FILE 
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d_%Y__%H_%M_%S')}.log"

# LOG FILE DIRECTORY AND NAME
LOG_FILE_DIR = os.path.join(os.getcwd(), "Logs")

# CREATE FOLDER IF NOT EXISTS
os.makedirs(LOG_FILE_DIR, exist_ok = True)

# LOG FILE PATH
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

logging.basicConfig(filename= LOG_FILE_PATH,
                    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s ",
                    level= logging.INFO)