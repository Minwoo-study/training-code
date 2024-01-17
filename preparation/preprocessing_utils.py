import os
import logging
import os


def ensure_log_dir(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)


def setup_logging(log_path):
    ensure_log_dir(log_path)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    all_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(all_format)
    f_handler.setFormatter(all_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def reconstruct_command(args, cmd): 
    for arg in vars(args):
        cmd += f" --{arg} {getattr(args, arg)}"
    return cmd
