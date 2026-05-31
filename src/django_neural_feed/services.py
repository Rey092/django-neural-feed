import numpy as np
from django.contrib.contenttypes.models import ContentType
from pgvector.django import CosineDistance
from django.db.models import F

from django_neural_feed.conf import app_settings


class RecommendationService:
    """
    Main business-logic engine responsible for mathematical vector modifications
    and fast database-level recommendation assembly.
    """

    _model_instance = None

    @classmethod
    def _get_model(cls):
        """Lazy loading wrapper to instantiate the heavy neural encoder model only when required."""
        if cls._model_instance is None:
            from sentence_transformers import SentenceTransformer

            cls._model_instance = SentenceTransformer(app_settings.MODEL_NAME)
        return cls._model_instance

    @classmethod
    def calculate_embedding(cls, text: str) -> list[float]:
        """Encodes raw text into a dense vector using the configured transformer."""
        model = cls._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @classmethod
    def calculate_user_embedding(
        cls, likes_queryset, content_field_name: str | None = None
    ) -> list[float] | None:
        """
        Builds a single user preference vector by computing the geometric mean
        of their most recently interacted content vectors.
        """
        limit = app_settings.USER_LIKES_LIMIT
        prefix = f"{content_field_name}__" if content_field_name else ""

        # Pull only target fields that already possess computed embeddings
        filter_kwargs = {f"{prefix}embedding__isnull": False}
        values_field = f"{prefix}embedding"

        # NOTE: Assumes the interaction model uses an incrementing 'id' for chronological ordering
        recent_emb = (
            likes_queryset.filter(**filter_kwargs)
            .order_by("-id")[:limit]
            .values_list(values_field, flat=True)
        )

        if not recent_emb:
            return None

        # Process vector averaging natively through numpy
        vectors_array = np.array(list(recent_emb))
        mean_vector = np.mean(vectors_array, axis=0)
        return mean_vector.tolist()

    @classmethod
    def get_feed_for_user(
        cls,
        user,
        model_class,
        queryset,
        likes_queryset,
        excluded_ids=None,
        limit: int = 20,
    ):
        """
        Compiles, values, and filters a lazy content QuerySet tailored to the user's vector signature
        using single-query multi-criteria annotations.
        """
        # Efficient database-level pruning of blacklisted or viewed IDs
        if excluded_ids is not None:
            queryset = queryset.exclude(id__in=excluded_ids)

        user_profile_vector = user.user_embedding

        # Graceful chronological fallback if the user has no history data
        if user_profile_vector is None:
            return queryset.order_by("-id")[:limit]

        # Multi-variable scoring calculation running natively on the DB instance via pgvector
        queryset = queryset.annotate(
            similarity=1 - CosineDistance("embedding", user_profile_vector),
            popularity=app_settings.POPULARITY_EXPRESSION,
            freshness=app_settings.FRESHNESS_EXPRESSION,
        )

        queryset = queryset.annotate(
            score=app_settings.WEIGHT_SIMILARITY * F("similarity")
            + app_settings.WEIGHT_FRESHNESS * F("freshness")
            + app_settings.WEIGHT_POPULARITY * F("popularity")
        ).order_by("-score")

        return queryset[:limit]
