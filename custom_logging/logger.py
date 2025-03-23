
import logging
import os

def setup_logger(name, log_folder, log_file):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(log_folder, log_file)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger

def get_logger(category="general"):
    # Each category will have its own folder and log file
    log_folder = os.path.join("logs", category)
    log_file = f"{category}.log"
    return setup_logger(f"{category}_logger", log_folder, log_file)

if __name__ == "__main__":
    logger = get_logger("training_log")
    logger.info("Training log initialized.")
