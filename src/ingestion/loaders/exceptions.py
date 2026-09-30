# src/ingestion/loaders/exceptions.py
"""Exceptions raised by the document loaders."""


class LoaderError(Exception):
    """Base class for all loader failures."""


class UnsupportedFormatError(LoaderError):
    """The file extension has no registered loader."""

    def __init__(self, extension: str, supported: list[str]):
        self.extension = extension
        self.supported = supported
        super().__init__(
            f"Unsupported file format: '{extension}'. "
            f"Supported formats: {', '.join(sorted(supported))}"
        )


class MissingDependencyError(LoaderError):
    """An optional dependency required for this format is not installed."""

    def __init__(self, feature: str, package: str, hint: str = ""):
        self.feature = feature
        self.package = package
        message = (
            f"Feature '{feature}' requires the '{package}' package, "
            f"which is not installed."
        )
        if hint:
            message = f"{message} {hint}"
        super().__init__(message)


class OcrNotAvailableError(MissingDependencyError):
    """OCR was needed but no OCR backend is usable."""


class ScannedDocumentError(LoaderError):
    """The document appears to be scanned but OCR is unavailable."""
