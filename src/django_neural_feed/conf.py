from django.conf import settings
from django.db.models import F, Case, When, Value, FloatField, ExpressionWrapper
from django.db.models.functions import Ln, Cast
from django.utils import timezone
from datetime import timedelta

# If developer didn't specify configs, we put default ones.
DEFAULT_CONFIG = {
    "MODEL_NAME": "intfloat/multilingual-e5-small", 
    "VECTOR_DIMENSION": 384,
    "WEIGHT_SIMILARITY": 0.6,
    "WEIGHT_FRESHNESS": 0.2,
    "WEIGHT_POPULARITY": 0.2,
}

def get_default_freshness_expression():
    """Returns freshness expression evaluated at query time."""
    return Case(
            When(
                created_at__gte=timezone.now() - timedelta(days=30),
                then=ExpressionWrapper(
                    (timezone.now() - F('created_at')) / Value(timedelta(days=30).total_seconds()),
                    output_field=FloatField()
                )
            ),
            output_field=FloatField(),
            default=Value(0.0)
        )

def get_default_popularity_expression():
    """Returns popularity expression."""
    raw_score = F('likes_count') + F('comments_count') * 2
    return Ln(Cast(raw_score, FloatField()) + 1.0)

class AppSettings:
    """Middle class for safe access to lib's functions."""
    
    def __init__(self):
        self._user_config = getattr(settings, "NEURAL_FEED_CONFIG", {})

    def _get_setting(self, key): 
        return self._user_config.get(key, DEFAULT_CONFIG[key])

    @property
    def MODEL_NAME(self) -> str:
        return self._get_setting("MODEL_NAME")  # HuggingFace model for text vectorization

    @property
    def VECTOR_DIMENSION(self) -> int:
        return self._get_setting("VECTOR_DIMENSION")  # Size of the embedding vector (e.g., 384 for E5-small)

    @property
    def WEIGHT_SIMILARITY(self) -> float:
        return self._get_setting("WEIGHT_SIMILARITY")  # Importance of semantic match with user interests in scoring formula

    @property
    def WEIGHT_FRESHNESS(self) -> float:
        return self._get_setting("WEIGHT_FRESHNESS")  # Importance of post creation time (recency) in scoring formula

    @property
    def WEIGHT_POPULARITY(self) -> float:
        return self._get_setting("WEIGHT_POPULARITY")  # Importance of post engagement (likes/views) in scoring formula

    @property
    def FRESHNESS_EXPRESSION(self):
        expr = self._user_config.get("FRESHNESS_EXPRESSION") # Django ORM expression for calculating freshness score (0-1)
        return expr if expr is not None else get_default_freshness_expression()

    @property
    def POPULARITY_EXPRESSION(self):
        expr = self._user_config.get("POPULARITY_EXPRESSION") # Django ORM expression for calculating popularity score
        return expr if expr is not None else get_default_popularity_expression()
    
    
app_settings = AppSettings()