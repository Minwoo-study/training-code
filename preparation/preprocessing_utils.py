import os
import logging
import os


def ensure_log_dir(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)


def setup_logging(log_path):
    ensure_log_dir(log_path)
    
    logging.basicConfig(filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger()


def reconstruct_command(args, cmd): 
    for arg in vars(args):
        cmd += f" --{arg} {getattr(args, arg)}"
    return cmd
