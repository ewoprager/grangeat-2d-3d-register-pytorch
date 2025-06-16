import logs_setup
from registration.lib import test_geometry

if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    test_geometry.test_plane_integrals()
