"""Contains all toolbox-specific exceptions that could be raised."""


class NotInCacheError(FileNotFoundError):

    """Exception raised when trying to load a slice of DAS data that has
    not been cached on disk."""

    def __init__(self, *args):
        """Construct the exception."""
        FileNotFoundError.__init__(self, *args)


class NoSuchPreambleError(FileNotFoundError):

    """Exception raised when trying to load a nonexistent predefined
    preamble."""

    def __init__(self, *args):
        """Construct the exception."""
        FileNotFoundError.__init__(self, *args)


class CSNotSupportedError(ValueError):

    """Exception raised when requesting to use an unsupported coordinate
    system."""

    def __init__(self, *args):
        """Construct the exception."""
        ValueError.__init__(self, *args)


class TrialDataExistsError(ValueError):

    """Exception raised when trying to register trial data for a date that
    already has such data."""

    def __init__(self, *args):
        """Construct the exception."""
        ValueError.__init__(self, *args)
