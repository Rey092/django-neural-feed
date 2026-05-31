from celery import shared_task
from django.apps import apps
from . import mixins


@shared_task
def generate_content_embedding_task(instance_id, model_path):
    try:
        app_label, model_name = model_path.split(".")

        model_class = apps.get_model(app_label, model_name)

        instance: mixins.NeuralRecommendMixin = model_class.objects.get(id=instance_id)  # type: ignore

        text_to_vectorize = instance.get_ready_text()

        if text_to_vectorize:
            from django_neural_feed.services import RecommendationService

            vector = RecommendationService.calculate_embedding(text_to_vectorize)

            instance.embedding = vector
            instance.save(update_fields=["embedding"])

    except Exception as e:  # if something went wrong, we logging it
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error (Celery) - can't generate embedding: {e}")
