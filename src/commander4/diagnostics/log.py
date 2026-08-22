"""Logging setup and the assert helpers that route failures through the logger.

`init_loggers` configures per-rank console and file handlers. `logassert`/`lograise` exist so that a
failed check is written to the log file (and identifies the rank) before the exception propagates,
which a bare `assert` in an MPI run would not do.
"""
import logging
import logging.config

# A recommended user setup is SUMMARY or INFO on the console and VERBOSE in the log file, so the
# file retains detailed sampling and performance diagnostics without overwhelming terminal output.

# --- Debug levels and their numeric values, including the Python defaults (commented out) ---
# DEBUG (10)    # Developer-only diagnostics that may be useful while troubleshooting.

VERBOSE = 15    # Detailed normal progress, including performance reports, solver checkpoints and
                # per-band diagnostics that are useful when investigating a run.

# INFO (20)     # Short scientific summary per sampling step; SUMMARY is once per Gibbs iteration.

SUMMARY = 25    # One compact end-of-Gibbs-iteration summary. Long stable runs are expected to use
                # this console level; WARNING and above suppresses all routine progress reporting.

# WARNING (30)  # Something might be wrong and should be checked by the user.

# ERROR (40)    # Things that should not happen. Note that this is still only a message, and will
                # not on its own abort the run. This is intentional: If a sensible fallback is
                # implemented, the run can continue even though something went wrong.

# CRITICAL (50) # Not currently used in Commander4.

# Introducing the two new levels of verbosity, along with the Python defaults.
logging.SUMMARY = SUMMARY
logging.VERBOSE = VERBOSE

logging.addLevelName(SUMMARY, "SUMMARY")
logging.addLevelName(VERBOSE, "VERBOSE")


def _summary(self, message, *args, **kwargs):
    if self.isEnabledFor(SUMMARY):
        self._log(SUMMARY, message, args, **kwargs)


def _verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


logging.Logger.summary = _summary
logging.Logger.verbose = _verbose


class C4Formatter(logging.Formatter):
    """Formatter that strips the leading 'commander4.' from logger names, so that
    e.g. 'commander4.tod.processing' is displayed as just 'tod_processing'."""

    def format(self, record: logging.LogRecord) -> str:
        record.name = record.name.removeprefix('commander4.')
        return super().format(record)


class WorldRankFilter(logging.Filter):
    """Attach the MPI world rank supplied during logger initialization to each record."""

    def __init__(self, world_rank: int | None):
        super().__init__()
        self.world_rank = "-" if world_rank is None else world_rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.world_rank = self.world_rank
        return True


class ColorFormatter(C4Formatter):
    """
    Logging formatter that prepends ANSI color codes to console output based on
    log level.

    Color scheme:
        DEBUG    - dim/faint (subtle, lower visual weight)
        VERBOSE  - dark gray
        INFO     - default terminal color
        SUMMARY  - default terminal color, bold
        WARNING  - yellow
        ERROR    - red
        CRITICAL - bold red
    """

    _RESET  = '\033[0m'
    _COLORS = {
        logging.DEBUG:    '\033[2m',    # Dim / faint
        VERBOSE:          '\033[90m',   # Dark gray (bright-black)
        logging.INFO:     '',           # Default, no color code
        SUMMARY:          '\033[1m',    # Default + bold
        logging.WARNING:  '\033[1;33m', # Yellow + bold
        logging.ERROR:    '\033[1;31m', # Red + bold
        logging.CRITICAL: '\033[1;31m', # Red + bold
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self._COLORS.get(record.levelno, '')
        return f'{color}{message}{self._RESET}' if color else message


def init_loggers(logger_params, log_file_path: str | None = None,
                 world_rank: int | None = None):
    """
    Intended usage: This function is called once at the very beginning of the
    program; after that, any function can call
    logger = logging.getLogger(__name__) to get a fully configured logger. This
    logger has debug(), verbose(), info(), summary(), warning(), error() and
    critical() logging functions, to be used as appropriate.

    Input arguments:
        logger_params (pixell.bunch): The part of the parameters that deal with
            logging. Each entry configures a separate logger, and currently,
            the two entry names supported are
                console: Needs a 'level' entry which can be one of the seven
                    logging levels (in lower or upper case)
                file: Needs a 'level' parameter like console.
        log_file_path (str): Where the file logger writes. The caller resolves it
            (see output.paths.log_file_path, which places the configured bare file
            name inside the run's logs directory), because the directory has to
            exist before the file handler here opens it.
        world_rank: MPI world rank included in every record. ``None`` is used by non-MPI callers.
    """

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
                'format': ('{asctime} - rank {world_rank:>3} - {name} - {levelname} - '
                           '{message}'),
                'datefmt': '%H:%M:%S',
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
        if log_file_path is None:
            raise ValueError("A 'file' logger is configured but no log_file_path was given.")
        config_dict['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': logger_params.file.level.upper(),
            'filename': log_file_path,
            'mode': 'a'
        }
        # Attach handler to root
        config_dict['loggers'][None]['handlers'].append('file')

    logging.config.dictConfig(config_dict)

    # Apply ColorFormatter to the console handler, and the (non-color) C4Formatter
    # to the file handler, both manually. This is also where prefix-stripping is
    # wired in for the file handler, since dictConfig's 'standard' formatter is a
    # plain logging.Formatter. dictConfig's '()' factory passes keyword arguments
    # directly to the constructor, but logging.Formatter.__init__ uses 'fmt' not
    # 'format', so the factory path silently falls back to an unconfigured
    # formatter; setting the formatters here avoids that problem entirely.
    _fmt = ('{asctime} - rank {world_rank:>3} - {name} - {levelname} - '
            '{message}')
    _datefmt = '%H:%M:%S'
    rank_filter = WorldRankFilter(world_rank)
    for _handler in logging.root.handlers:
        _handler.addFilter(rank_filter)
        if isinstance(_handler, logging.FileHandler):
            _handler.setFormatter(C4Formatter(fmt=_fmt, datefmt=_datefmt, style='{'))
        elif isinstance(_handler, logging.StreamHandler):
            _handler.setFormatter(ColorFormatter(fmt=_fmt, datefmt=_datefmt, style='{'))

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
