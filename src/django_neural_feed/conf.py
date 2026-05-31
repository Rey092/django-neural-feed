from django.conf import settings
from django.db.models import F, Case, When, Value, FloatField, ExpressionWrapper
from django.db.models.functions import Ln, Cast
from django.utils import timezone
from datetime import timedelta
from .exceptions import ImproperlyConfigured

# If developer didn't specify configs, we put default ones.
DEFAULT_CONFIG = {
    "MODEL_NAME": "intfloat/multilingual-e5-small",
    "VECTOR_DIMENSION": 384,
    "WEIGHT_SIMILARITY": 0.6,
    "WEIGHT_FRESHNESS": 0.2,
    "WEIGHT_POPULARITY": 0.2,

    "CELERY_ENABLED": False,
}


class AppSettings:
    """Middle class for safe access to lib's functions."""

    def __init__(self):
        self._user_config = getattr(settings, "NEURAL_FEED_CONFIG", {})

    def _get_setting(self, key):
        return self._user_config.get(key, DEFAULT_CONFIG[key])

    @property
    def MODEL_NAME(self) -> str:
        return self._get_setting(
            "MODEL_NAME"
        )  # HuggingFace model for text vectorization

    @property
    def VECTOR_DIMENSION(self) -> int:
        return self._get_setting(
            "VECTOR_DIMENSION"
        )  # Size of the embedding vector (e.g., 384 for E5-small)

    @property
    def WEIGHT_SIMILARITY(self) -> float:
        return self._get_setting(
            "WEIGHT_SIMILARITY"
        )  # Importance of semantic match with user interests in scoring formula

    @property
    def WEIGHT_FRESHNESS(self) -> float:
        return self._get_setting(
            "WEIGHT_FRESHNESS"
        )  # Importance of post creation time (recency) in scoring formula

    @property
    def WEIGHT_POPULARITY(self) -> float:
        return self._get_setting(
            "WEIGHT_POPULARITY"
        )  # Importance of post engagement (likes/views) in scoring formula

    @property
    def FRESHNESS_EXPRESSION(self):
        expr = self._user_config.get(
            "FRESHNESS_EXPRESSION"
        )  # Django ORM expression for calculating freshness score (0-1)
        if expr is not None:
            return expr
        else:
            raise ImproperlyConfigured("FRESHNESS_EXPRESSION wasn't configured!")

    @property
    def POPULARITY_EXPRESSION(self):
        expr = self._user_config.get(
            "POPULARITY_EXPRESSION"
        )  # Django ORM expression for calculating popularity score
        if expr is not None:
            return expr
        else:
            raise ImproperlyConfigured("POPULARITY_EXPRESSION wasn't configured!")
        
    @property
    def CELERY_ENABLED(self) -> bool:
        return self._get_setting(
            "CELERY_ENABLED"
        )  # Are we using Celery?


app_settings = AppSettings()
