import numpy as np
from django.db.models import F, Value
from django.db.models.functions import Coalesce
from pgvector.django import MaxInnerProduct

from django_neural_feed.conf import app_settings


class BaseNeuralFeed:

    feed_id: str = "default_feed"
    content_django_model = None
    interaction_django_model = None
    user_field_name: str | None = None
    content_field_name: str | None = None
    parent_feed: type["BaseNeuralFeed"] | None = None

    embedding_model_name: str = app_settings.MODEL_NAME
    user_likes_limit: int = app_settings.USER_LIKES_LIMIT

    weight_similarity: float = app_settings.WEIGHT_SIMILARITY
    weight_freshness: float = app_settings.WEIGHT_FRESHNESS
    weight_popularity: float = app_settings.WEIGHT_POPULARITY

    popularity_expression = app_settings.POPULARITY_EXPRESSION
    freshness_expression = app_settings.FRESHNESS_EXPRESSION

    _model_instances: dict = {}

    @classmethod
    def get_setting(cls, attr_name: str):
        """
        Settings getter with fallback to parent_feed before the base class defaults.
        """
        if attr_name in cls.__dict__:
            return getattr(cls, attr_name)

        if cls.parent_feed is not None:
            return cls.parent_feed.get_setting(attr_name)

        return getattr(cls, attr_name)

    @classmethod
    def calculate_embedding(cls, text: str) -> list[float]:
        encoder = app_settings.ENCODER_CLASS
        embedding = encoder.text_to_vector(
            text, cls.get_setting("embedding_model_name")
        )
        return embedding

    @classmethod
    def calculate_user_embedding(
        cls, likes_queryset, content_field_name: str | None = None
    ) -> list[float] | None:
        limit = cls.get_setting("user_likes_limit")
        prefix = f"{content_field_name}__" if content_field_name else ""

        filter_kwargs = {f"{prefix}embedding__isnull": False}
        values_field = f"{prefix}embedding"

        recent_emb = list(
            likes_queryset.filter(**filter_kwargs)
            .order_by("-id")[:limit]
            .values_list(values_field, flat=True)
        )

        if not recent_emb:
            return None

        vectors_array = np.asarray(recent_emb, dtype=np.float32)
        mean_vector = np.mean(vectors_array, axis=0)

        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        return mean_vector.tolist()

    @classmethod
    def get_user_vector(cls, user) -> list[float] | None:
        """
        Retrieves the averaged interaction vector for a specific user
        scoped to this particular feed instance.
        """
        if user is None or not user.is_authenticated:
            return None

        from django_neural_feed.models import UserFeedProfile

        target_feed_id = cls.get_setting("feed_id")

        try:
            profile = UserFeedProfile.objects.filter(
                user_id=user.id, feed_id=target_feed_id
            ).first()

            return profile.embedding if profile else None

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"DNF: Error fetching user vector for feed '{target_feed_id}': {e}"
            )
            return None

    @classmethod
    def get_candidates(cls, user, queryset, excluded_ids=None):
        if queryset is None:
            queryset = cls.get_setting("content_django_model").objects.all()  # type: ignore

        if excluded_ids is not None:
            queryset = queryset.exclude(id__in=excluded_ids)

        return queryset

    @classmethod
    def rank_candidates(cls, queryset, user_profile_vector):
        queryset = queryset.annotate(
            similarity=Coalesce(
                -MaxInnerProduct("embedding", user_profile_vector), Value(0.0)
            ),
            popularity=Coalesce(cls.popularity_expression, Value(0.0)),
            freshness=Coalesce(cls.freshness_expression, Value(0.0)),
        )

        queryset = queryset.annotate(
            score=cls.weight_similarity * F("similarity")
            + cls.weight_freshness * F("freshness")
            + cls.weight_popularity * F("popularity")
        ).order_by("-score")

        return queryset

    @classmethod
    def get_feed(cls, user, queryset=None, excluded_ids=None, limit: int = 20):

        candidates_qs = cls.get_candidates(user, queryset, excluded_ids)

        user_profile_vector = cls.get_user_vector(user)

        if user_profile_vector is None:
            return candidates_qs.order_by("-id")[:limit]

        ranked_qs = cls.rank_candidates(candidates_qs, user_profile_vector)

        return ranked_qs[:limit]
