# Keep this file free of imports. `TODSamples` imports `events.py`, which runs this file first, and
# `sampling.py` imports `TODView`, which imports `TODSamples`. Importing `sampling` here would close
# that loop and fail at startup.
