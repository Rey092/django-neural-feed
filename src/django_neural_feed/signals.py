from django.db.models.signals import post_save
from django.dispatch import receiver
from django_neural_feed.mixins import NeuralRecommendMixin
from django_neural_feed.conf import app_settings


@receiver(post_save)
def generate_content_embedding(sender, instance, created, **kwargs):
    """On model save signal."""

    if not issubclass(sender, NeuralRecommendMixin):
        return

    update_fields = kwargs.get("update_fields")
    if (
        update_fields and "embedding" in update_fields
    ):  # Prevents infinite loop. When we save a new embedding, it triggers signal too!
        return

    should_generate = (
        created or instance.embedding is None or (update_fields is not None)
    )  # If post just got created, or we didn't put embedding, we should generate embedding!

    if should_generate:
        if app_settings.CELERY_ENABLED:
            try:
                from celery import Task
                from .tasks import generate_content_embedding_task

                model_path = f"{sender._meta.app_label}.{sender._meta.model_name}"
                celery_task: Task = generate_content_embedding_task  # type: ignore

                celery_task.delay(instance.id, model_path)
                return

            except (ImportError, ModuleNotFoundError):
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "DNF Config: CELERY_ENABLED is True, but celery is not installed! "
                    "Falling back to synchronous embedding generation."
                )

        try:
            text_to_vectorize = (
                instance.get_ready_text()
            )  # get_ready_text function should be assigned by developer

            if text_to_vectorize:
                from django_neural_feed.services import RecommendationService

                vector = RecommendationService.calculate_embedding(text_to_vectorize)

                instance.embedding = vector
                instance.save(update_fields=["embedding"])

        except Exception as e:  # if something went wrong, we logging it
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Error - can't generate embedding for {sender.__name__} (id: {instance.pk}): {e}"
            )
