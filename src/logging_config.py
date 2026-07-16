import logging
import os
from logging.handlers import RotatingFileHandler

# put logs in a logs/ folder at project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")


def setup_logging():
    """Configure logging to write to BOTH the console and a rotating file.
    Call this once when the app starts."""
    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.INFO)

    # avoid adding duplicate handlers if this runs more than once
    if logger.handlers:
        return logger

    # format: timestamp - level - message
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. console handler — so you see logs live while developing
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 2. rotating file handler — persistent record, caps file size so
    #    logs don't grow forever (5MB per file, keep 3 old files)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# a module-level logger other files can import
logger = setup_logging()
