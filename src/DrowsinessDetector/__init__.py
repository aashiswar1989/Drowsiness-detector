import logging
from pathlib import Path
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
LOG_DIR = Path('logs')

if not LOG_DIR.is_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR/LOG_FILE

logging.basicConfig(
    level =  logging.INFO,
    filename = LOG_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully.")