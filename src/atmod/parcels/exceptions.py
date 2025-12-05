"""Custom exceptions for parcel processing."""


class ParcelError(Exception):
    """Base exception for parcel-related errors."""

    pass


class ParcelExtractionError(ParcelError):
    """Error during parcel subsurface extraction."""

    pass


class ParcelOutsideBoundsError(ParcelExtractionError):
    """Parcel centroid is outside the subsurface model bounds."""

    pass


class NoSubsurfaceDataError(ParcelExtractionError):
    """No valid subsurface data at parcel location."""

    pass


class InvalidGeometryError(ParcelExtractionError):
    """Parcel geometry is invalid and cannot be repaired."""

    pass


class ValidationError(ParcelError):
    """Input validation failed."""

    pass


class DuplicateParcelIdError(ValidationError):
    """Duplicate parcel IDs found in input."""

    pass
