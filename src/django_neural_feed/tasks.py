from celery import shared_task
from django.apps import apps
from .services import RecommendationService
from django.db.models import Model
from . import mixins


def get_model_from_path(model_path: str) -> type[Model] | None:
    """
    Helper function to dynamically look up a Django model class using
    its 'app_label.model_name' identifier string.
    """
    model_class = None
    try:
        app_label, model_name = model_path.split(".")
        model_class = apps.get_model(app_label, model_name)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error (Celery) - can't get model from path ({model_path}): {e}")

    return model_class


@shared_task
def generate_content_embedding_task(instance_id, model_path):
    """Asynchronously calculates and saves vector embeddings for content models."""
    try:
        model_class = get_model_from_path(model_path)
        if model_class is None:
            return

        instance: mixins.NeuralRecommendMixin = model_class.objects.get(id=instance_id)  # type: ignore
        text_to_vectorize = instance.get_ready_text()

        if text_to_vectorize:
            vector = RecommendationService.calculate_embedding(text_to_vectorize)
            instance.embedding = vector
            instance.save(update_fields=["embedding"])

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error (Celery) - can't generate embedding: {e}")


@shared_task
def update_user_embedding_task(
    likes_model_path, users_model_path, user_id, user_field_name, content_field_name
):
    """Asynchronously recalculates the user profile vector based on recent interaction history."""
    try:
        likes_model = get_model_from_path(likes_model_path)
        user_model = get_model_from_path(users_model_path)

        if likes_model is None or user_model is None:
            return

        # Safe lookup in case the user was purged from the DB while the job was queued
        try:
            user_object: mixins.NeuralUserMixin = user_model.objects.get(id=user_id)  # type: ignore
        except user_model.DoesNotExist:
            return

        filter_kwargs = {f"{user_field_name}_id": user_id}
        likes_queryset = likes_model.objects.filter(**filter_kwargs)

        vector = RecommendationService.calculate_user_embedding(
            likes_queryset, content_field_name
        )
        user_object.user_embedding = vector
        user_object.save(update_fields=["user_embedding"])

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error (Celery) - can't generate user embedding: {e}")
