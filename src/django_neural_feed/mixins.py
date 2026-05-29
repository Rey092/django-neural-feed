from django.db import models
from pgvector.django import VectorField

from django_neural_feed.conf import app_settings


class NeuralRecommendMixin(models.Model):
    """Mixin for recommendable items
    """

    embedding = VectorField(
        dimensions=app_settings.VECTOR_DIMENSION, null=True, blank=True
    )

    class Meta:
        abstract = True

    def get_ready_text(self) -> str:
        """String interpretation of model. (e.g. for post it's a text)
        """
        raise NotImplementedError(
            "You should assign get_ready_text() in your model!"
        )
    
class NeuralUserMixin(models.Model):
    neural_vector = VectorField(dimensions=app_settings.VECTOR_DIMENSION, null=True, blank=True)