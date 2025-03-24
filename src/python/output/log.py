import logging
import logging.config


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

    # This part is to configure the healpy logger, and should be relatively independent of the main logger stuff
    hp_logger = logging.getLogger("healpy")
    hp_logger.setLevel(logging.WARNING)
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
            None: { # Here we can potentially add more loggers, with custom names.
                'handlers': [],
                'level': 'DEBUG',
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
        config_dict['loggers'][None]['handlers'].append('console')
    if 'file' in logger_params:
        config_dict['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': logger_params.file.level.upper(),
            'filename': logger_params.file.filename,
            'mode': 'a'
        }
        config_dict['loggers'][None]['handlers'].append('file')
    logging.config.dictConfig(config_dict)


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
