import logging
import logging.config

# --- Debug levels and their numeric values, including the Python defaults (commented out) ---
# DEBUG (10)    # >100 per iter, intermediate processing steps and per-rank or per-detector details.
VERBOSE = 15    # ~30 per iter, print per-band results and performance.
# INFO (20)     # ~10 per iter, print high-level results and high-level runtime summary.
QUIET = 25      # < 5 per iter, short iteration summary.
# WARNING (30)
# ERROR (40)
# CRITICAL (50)

logging.QUIET = QUIET
logging.VERBOSE = VERBOSE


def init_loggers(logger_params): 
    """
    Intended usage: This function is called once at the very beginning of the
    program; after that, any function can call
    logger = logging.getLogger(__name__) to get a fully configured logger. This
    logger has debug(), info(), warning(), error() and critical() logging
    functions, to be used as appropriate.

    Input arguments:
        logger_params (pixell.bunch): The part of the parameters that deal with
            logging. Each entry configures a separate logger, and currently,
            the two entry names supported are
                console: Needs a 'level' entry which can be one of the five
                    logging levels (in lower or upper case)
                file: Needs a 'level' parameter like console, as well as a
                    'filename' entry, giving the output file name.
    """

    # --- ADDITIONAL VERBOSITY LAYERS ---
    # In addition to the default logging verbosity levels, add two more levels.
    logging.addLevelName(QUIET, "QUIET")
    logging.addLevelName(VERBOSE, "VERBOSE")

    def quiet(self, message, *args, **kws):
        if self.isEnabledFor(QUIET):
            self._log(QUIET, message, args, **kws)
    def verbose(self, message, *args, **kws):
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, message, args, **kws)

    logging.Logger.verbose = verbose
    logging.Logger.quiet = quiet

    # --- DETETERMINE LOWEST LOGGING LEVEL ---
    # Determine the most verbose logging level among the loggers, so that we capture all loggings
    # up to this level, before more detailed filtering in the specific loggers.
    levels = []
    if 'console' in logger_params: levels.append(logger_params.console.level.upper())
    if 'file' in logger_params: levels.append(logger_params.file.level.upper())
    # Helper to resolve "DEBUG", "INFO" strings to integers for comparison
    def level_val(name): return getattr(logging, name, 20)
    # Find the minimum level (most verbose) requested
    min_level_name = 'INFO'  # Default to INFO if no levels exist.
    if levels:
        min_level_name = min(levels, key=level_val)

    # --- CONFIGURE LOGGERS ---
    config_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'style': '{',
                'format': '{asctime} - {name} - {levelname} - {message}'
            },
        },
        'handlers': {},
        'loggers': {
            # The Root Logger (None) is set to WARNING. This means that packages like h5py or pysm3
            # will only log WARNINGs, and not e.g. INFO or DEBUG if such levels are set.
            None: { 
                'handlers': [],
                'level': 'WARNING',  # Silence third-party noise
                'propagate': True
            },
            # The Commander4 logger, taking the actually requested verbosity level.
            'commander4': {
                'handlers': [],
                'level': min_level_name, # Maximum verbosity of 'file' and 'console'.
                'propagate': True
            }
        }
    }
    if 'console' in logger_params:
        config_dict['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': logger_params.console.level.upper()
        }
        # Attach handler to root so it captures everything that passes the filter
        config_dict['loggers'][None]['handlers'].append('console')

    if 'file' in logger_params:
        config_dict['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': logger_params.file.level.upper(),
            'filename': logger_params.file.filename,
            'mode': 'a'
        }
        # Attach handler to root
        config_dict['loggers'][None]['handlers'].append('file')

    logging.config.dictConfig(config_dict)

    # Configure logging to redirect warnings from py.warning. Note that this will *prevent* these
    # from being sent to sys.stderr, to avoid duplication.
    logging.captureWarnings(True)


def logassert(assertion, errmsg, logger):
    "Asserts and prints to logger if false"
    try:
        assert assertion
    except AssertionError as err:
        lograise(err, errmsg, logger)


def logassert_np(assertion, errmsg, logger):
    "Asserts (using numpy.testing.assert_) and prints to logger if false"
    from numpy.testing import assert_ as myassert
    try:
        myassert(assertion, '')
    except AssertionError as err:
        lograise(err, errmsg, logger)


def lograise(error, errmsg, logger):
    "Prints to logger and raises an error"

    logger.exception(errmsg)
    raise error
