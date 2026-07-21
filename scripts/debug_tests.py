from reg23_experiments.registration.lib import test_geometry
from reg23_experiments.utils import logs_setup

if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    test_geometry.test_plane_integrals()
