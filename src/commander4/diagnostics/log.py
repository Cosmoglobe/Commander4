"""Commander4 logging setup and fatal MPI-rank failure reporting."""
import logging
import logging.config
import os
import secrets
import string
import sys
import time
import traceback
from datetime import datetime

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


def make_run_id() -> str:
    """Return a readable, high-precision run identifier with a short random suffix."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f%z")
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(4))
    return f"{timestamp}-{suffix}"


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
        logging.DEBUG:    '\033[90m',   # Dark gray (bright-black)
        VERBOSE:          '\033[2m',    # Dim / faint
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
        # StreamHandler defaults to sys.stderr. In batch jobs this is normally collected by Slurm
        # together with stdout, but keeping it as stderr correctly marks logging as diagnostic
        # output and leaves stdout available for explicit program output.
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
        # The ordinary run log is shared by the MPI ranks. Fatal tracebacks are additionally
        # protected by the single-writer mechanism in report_fatal below.
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


def report_fatal(error: BaseException, logger: logging.Logger, world_rank: int,
                 error_code: int, traceback_dir: str | None = None,
                 context: str | None = None, run_id: str | None = None) -> str | None:
    """Preserve the first fatal traceback for this run before MPI aborts.

    A Python exception normally reaches this function on only the failing rank. Some shared errors
    can reach many ranks at almost the same time, however, and printing every traceback would flood
    both the terminal and the shared log. Ranks therefore compete to create one run-specific file.
    The operating system guarantees that ``O_EXCL`` lets only one rank create it. That owner writes
    the traceback and the only CRITICAL record; other ranks wait briefly and stay silent.

    ``stderr`` is the process's diagnostic text stream and is where the console logging handler
    writes. If logging or the output directory is unavailable, stderr is the last-resort location
    because Slurm normally captures it even when Commander4 cannot open its own files.

    Returns:
        The traceback path when one was written, otherwise ``None``.
    """
    error_name = type(error).__name__
    error_text = str(error)
    description = f"{error_name}: {error_text}" if error_text else error_name
    if context:
        description = f"{context}: {description}"

    # This marker tells non-owning ranks that both the traceback and normal log record were flushed.
    # They may then call MPI_Abort without risking termination of the owner halfway through a write.
    report_complete = "\n--- END FATAL REPORT ---\n"
    traceback_path = None
    traceback_write_error = None
    if traceback_dir is not None:
        report_name = f"fatal-{run_id}.log" if run_id else "fatal.log"
        traceback_path = os.path.join(traceback_dir, report_name)
        try:
            os.makedirs(traceback_dir, exist_ok=True)
            # O_EXCL makes creation atomic across ranks on the shared output filesystem: success
            # means this rank owns reporting; FileExistsError means another rank got there first.
            file_descriptor = os.open(
                traceback_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # The owning rank writes the CRITICAL record before the completion marker. Waiting for
            # that marker prevents another rank's immediate MPI_Abort from killing it mid-write.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    with open(traceback_path, "rb") as handle:
                        if handle.read().endswith(report_complete.encode("ascii")):
                            break
                except OSError:
                    pass
                time.sleep(0.02)
            logging.shutdown()
            return traceback_path
        except OSError as write_error:
            # A missing or broken output filesystem must not conceal the original exception.
            traceback_write_error = write_error
            traceback_path = None
        else:
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                    # First make the standalone traceback durable. This file remains readable even
                    # if MPI termination prevents the normal shared log from closing cleanly.
                    handle.write(f"\n{datetime.now().isoformat(timespec='seconds')} - world rank "
                                 f"{world_rank} - run {run_id or '-'} - abort code {error_code}\n")
                    handle.write(f"{description}\n")
                    traceback.print_exception(type(error), error, error.__traceback__, file=handle)
                    handle.flush()
                    os.fsync(handle.fileno())

                    # exc_info adds the standard Python traceback, including source file and line
                    # numbers, to both the console handler (stderr) and the ordinary run log.
                    message = (f"Fatal error on world rank {world_rank}: \n{description}; calling "
                               f"MPI Abort({error_code}). Full traceback: {traceback_path}")
                    if logging.root.handlers:
                        logger.critical(
                            message,
                            exc_info=(type(error), error, error.__traceback__),
                        )
                    else:
                        print(message, file=sys.stderr, flush=True)
                        traceback.print_exception(
                            type(error), error, error.__traceback__, file=sys.stderr)
                    # MPI_Abort does not guarantee normal Python cleanup, so explicitly flush and
                    # close logging handlers before allowing any rank to terminate the job.
                    logging.shutdown()

                    # Write the marker last. Non-owner ranks poll for it for at most two seconds;
                    # the timeout prevents a deadlock if the owner itself dies while reporting.
                    handle.write(report_complete)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as write_error:
                traceback_write_error = write_error
                traceback_path = None
            else:
                return traceback_path

    # Early startup failures may occur before the output path or handlers exist. They cannot use
    # atomic file election, so each affected rank falls back to stderr and preserves its traceback.
    message = (f"Fatal error on world rank {world_rank}: \n{description}; calling MPI "
               f"Abort({error_code})")
    if traceback_write_error is not None:
        message += f". Could not write the traceback file: {traceback_write_error}"
    else:
        message += ". Full traceback follows on stderr"

    if logging.root.handlers:
        logger.critical(message)
    else:
        print(message, file=sys.stderr, flush=True)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()
    logging.shutdown()
    return None
