import numpy as np
from django.contrib.contenttypes.models import ContentType
from pgvector.django import CosineDistance

from django_neural_feed.conf import app_settings


class RecommendationService:
    _model_instance = None

    @classmethod
    def _get_model(cls):
        """Lazy AI model initialization.
        """
        if cls._model_instance is None:
            from sentence_transformers import SentenceTransformer
            cls._model_instance = SentenceTransformer(app_settings.MODEL_NAME)
        return cls._model_instance

    @classmethod
    def calculate_embedding(cls, text: str) -> list[float]:
        """Transforms text into vectors."""
        model = cls._get_model()
        #generating numpy embedding
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    @classmethod
    def calculate_user_embedding(cls, likes_queryset, limit: int = 20):
        recent_emb = likes_queryset.filter(embedding__isnull=False).order_by('-id')[:limit].values_list('embedding', flat=True)

        if not recent_emb:
            return None
        
        vectors_array = np.array(list(recent_emb))
        mean_vector = np.mean(vectors_array,axis=0)
        return mean_vector.tolist()


    @classmethod
    def get_feed_for_user(cls, user, model_class, queryset, likes_queryset, limit: int = 20):
        """Main feed generation function
        """
        """
        # getting list of disliked objects for user
        disliked_ids = UserDislike.objects.filter(
            user=user,
            content_type=ContentType.objects.get_for_model(model_class)
        ).values_list('object_id', flat=True)

        # removing disliked posts from search
        queryset = queryset.exclude(id__in=disliked_ids) """
        
        user_profile_vector = cls.calculate_user_embedding(likes_queryset, limit)

        if user_profile_vector is None:
            return queryset.order_by('-id')[:limit]

        """ TODO:
        We have:
            "WEIGHT_SIMILARITY": 0.6, 60%
            "WEIGHT_FRESHNESS": 0.2, 20%
            "WEIGHT_POPULARITY": 0.2, 20%

        For example: from 10 recomended videos we will get:
        6 videos about topic we interested in,
        2 videos that was posted recently (we need somehow filter shit content but idk how lol)
        And 2 viral videos
        """
        queryset = queryset.annotate(
            distance=CosineDistance('embedding', user_profile_vector)
        ).order_by('distance')

        return queryset[:limit]