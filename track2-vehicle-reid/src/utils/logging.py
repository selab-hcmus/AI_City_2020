"""Global logger"""
import os
import logging


def setup_logging(log_fname=None):
    """Setup format for global logger

    Args:
        log_fname: filename to save the log to. The log will show on stdout and
            saved to that file. If nothing is given, the log will show on stdout
            only.
    """
    # Set up format
    # my_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # my_fmt = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    my_fmt = '[%(levelname)s: %(name)s]: %(message)s'

    # Set up file handlers
    if log_fname is not None:
        # Make dir if not exists
        parent_dir = os.path.dirname(log_fname)
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)

        my_handlers = [
            logging.FileHandler(log_fname),
            logging.StreamHandler(),
        ]
    else:
        my_handlers = [logging.StreamHandler()]

    # Change the log setup
    logging.basicConfig(
        level=logging.INFO,
        format=my_fmt,
        handlers=my_handlers,
    )


def get_logger(name):
    """Get the logger with a given name

    Args:
        name: name of the logger to create
    """
    return logging.getLogger(name)
