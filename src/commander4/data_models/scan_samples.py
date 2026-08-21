"""`ScanSamples`: a placeholder container, used only by the deprecated Planck sim reader."""

class ScanSamples:
    """Placeholder for one scan's sampled quantities.

    Only used by the legacy ``experiments/planck/sim`` reader; the live path carries per-scan
    samples in the dense ``TODSamples`` arrays instead.
    """

    def __init__(self):
        pass
