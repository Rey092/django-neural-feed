import threading
from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver
from django.db.models import Model
from django.db import connection, transaction
from django_neural_feed.conf import app_settings


@receiver(post_save)
def generate_content_embedding(sender, instance, created, **kwargs):
    """Autogeneration of embedding for models with NeuralRecommendMixin."""
    from django_neural_feed.mixins import NeuralRecommendMixin

    if not issubclass(sender, NeuralRecommendMixin):
        return

    update_fields = kwargs.get("update_fields")
    if update_fields and "embedding" in update_fields:
        return

    should_generate = (
        created or instance.embedding is None or (update_fields is not None)
    )

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

                logging.getLogger(__name__).warning(
                    "DNF: CELERY_ENABLED is True, but celery is not installed! Falling back to background threads."
                )

        transaction.on_commit(
            lambda: threading.Thread(
                target=_run_synchronous_content_update,
                args=(sender, instance.id),
                daemon=True,
            ).start()
        )


def register_like_signal(
    like_target,
    user_field_name: str | None = None,
    content_field_name: str | None = None,
):
    """
    Universal like register. Supports default tables and m2m
    """

    def user_like_changed_model(sender, instance, created, **kwargs):
        if created:
            try:
                user_object = getattr(instance, user_field_name)  # type: ignore
                _trigger_embedding_update(
                    user_object,
                    sender,
                    user_object.id,
                    user_field_name,
                    content_field_name,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(f"DNF Error (model signal): {e}")

    def user_like_changed_m2m(sender, instance, action, reverse, pk_set, **kwargs):
        if action == "post_add":
            try:
                from django.contrib.auth import get_user_model

                User = get_user_model()
                target_content_field = None
                target_user_field = None

                for field in sender._meta.fields:
                    if field.is_relation:
                        if issubclass(instance.__class__, field.related_model):
                            target_content_field = field.name
                        elif (
                            issubclass(User, field.related_model)
                            or field.related_model == User
                        ):
                            target_user_field = field.name

                if reverse:
                    user_object = instance
                    _trigger_embedding_update(
                        user_object,
                        sender,
                        user_object.id,
                        target_user_field,
                        target_content_field,
                    )
                else:
                    for user_id in pk_set:
                        user_object = User.objects.get(pk=user_id)
                        _trigger_embedding_update(
                            user_object,
                            sender,
                            user_id,
                            target_user_field,
                            target_content_field,
                        )
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(f"DNF Error (M2M signal): {e}")

    if hasattr(like_target, "_meta") and like_target._meta.auto_created:
        m2m_changed.connect(user_like_changed_m2m, sender=like_target)
    else:
        if not user_field_name or not content_field_name:
            raise ValueError(
                "For custom like model specify user_field_name and content_field_name"
            )
        post_save.connect(user_like_changed_model, sender=like_target)


def _trigger_embedding_update(
    user_object, sender, user_id, user_field_name, content_field_name
):
    if app_settings.CELERY_ENABLED:
        try:
            from celery import Task
            from .tasks import update_user_embedding_task

            likes_model_path = f"{sender._meta.app_label}.{sender._meta.model_name}"
            users_model_path = f"{user_object.__class__._meta.app_label}.{user_object.__class__._meta.model_name}"
            celery_task: Task = update_user_embedding_task  # type: ignore
            celery_task.delay(
                likes_model_path,
                users_model_path,
                user_id,
                user_field_name,
                content_field_name,
            )
            return
        except (ImportError, ModuleNotFoundError):
            pass

    transaction.on_commit(
        lambda: threading.Thread(
            target=_run_synchronous_user_update,
            args=(
                user_object.__class__,
                user_id,
                sender,
                user_field_name,
                content_field_name,
            ),
            daemon=True,
        ).start()
    )


def _run_synchronous_content_update(model_class, instance_id):
    """Background thread for content embedding calculation"""
    try:
        instance = model_class.objects.get(id=instance_id)
        text_to_vectorize = instance.get_ready_text()
        if text_to_vectorize:
            from django_neural_feed.services import RecommendationService

            instance.embedding = RecommendationService.calculate_embedding(
                text_to_vectorize
            )
            instance.save(update_fields=["embedding"])
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"DNF Background Thread Error (Content): {e}")
    finally:
        connection.close()


def _run_synchronous_user_update(
    user_model, user_id, sender_model, user_field_name, content_field_name
):
    """Background thread for user embedding calculation"""
    try:
        user_object = user_model.objects.get(id=user_id)
        filter_kwargs = {f"{user_field_name}_id": user_id}
        likes_queryset = sender_model.objects.filter(**filter_kwargs)

        from django_neural_feed.services import RecommendationService

        vector = RecommendationService.calculate_user_embedding(
            likes_queryset, content_field_name
        )
        user_object.user_embedding = vector
        user_object.save(update_fields=["user_embedding"])
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"DNF Background Thread Error (User): {e}")
    finally:
        connection.close()
