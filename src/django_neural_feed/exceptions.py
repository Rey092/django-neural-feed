class ImproperlyConfigured(Exception):
    def __init__(self, message="Check DNF configs, something is wrong."):
        super().__init__(message)
