import logging.config

from registration.lib import test_geometry

if __name__ == "__main__":
    # set up logger
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    logger = logging.getLogger("radonRegistration")

    test_geometry.test_plane_integrals()
