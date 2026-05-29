from django.conf import settings

# If developer didn't specify configs, we put default ones.
DEFAULT_CONFIG = {
    "MODEL_NAME": "intfloat/multilingual-e5-small", 
    "VECTOR_DIMENSION": 384,
    "WEIGHT_SIMILIARITY": 0.6,
    "WEIGHT_FRESHNESS": 0.2,
    "WEIGHT_POPULARITY": 0.2,
    "EXPLORE_RATIO": 0.15,
}

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
    def WEIGHT_SIMILIARITY(self) -> float:
        return self._get_setting("WEIGHT_SIMILIARITY")  # Importance of semantic match with user interests in scoring formula

    @property
    def WEIGHT_FRESHNESS(self) -> float:
        return self._get_setting("WEIGHT_FRESHNESS")  # Importance of post creation time (recency) in scoring formula

    @property
    def WEIGHT_POPULARITY(self) -> float:
        return self._get_setting("WEIGHT_POPULARITY")  # Importance of post engagement (likes/views) in scoring formula

    @property
    def EXPLORE_RATIO(self) -> float:
        return self._get_setting("EXPLORE_RATIO")  # Percentage of random/discovery content mixed into the final feed

app_settings = AppSettings()