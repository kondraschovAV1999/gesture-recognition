class ConfigError(Exception):
    """Base class for configuration errors."""


class ConfigValidationError(ConfigError):
    """Raised when a resolved config violates the schema or cross-field rules."""


class ConfigReplayError(ConfigError):
    """Raised when replaying a persisted config fails (missing/incompatible)."""
