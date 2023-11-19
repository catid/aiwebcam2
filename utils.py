import logging

import colorama
from colorama import Fore, Style

colorama.init()

class ColoredFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        if not record.exc_info:
            record.msg = f"{ColoredFormatter.level_colors[record.levelno]}{record.msg}{Style.RESET_ALL}"
        return super(ColoredFormatter, self).format(record)

def setup_colored_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

logger = setup_colored_logging()
