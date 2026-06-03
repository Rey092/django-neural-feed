import threading
from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver
from django.db import connection, transaction
from django_neural_feed.conf import app_settings
from typing import Literal
import logging

logger = logging.getLogger(__name__)


@receiver(post_save)
def generate_content_embedding(sender, instance, created, **kwargs):
    """Autogeneration of embedding for models with NeuralRecommendMixin."""
    from django_neural_feed.mixins import NeuralRecommendMixin

    if not issubclass(sender, NeuralRecommendMixin):
        return

    update_fields = kwargs.get("update_fields")
    if update_fields and "embedding" in update_fields:
        return

    text_changed = False
    if not created:
        current_text = instance.get_ready_text()
        text_changed = current_text != getattr(instance, "_original_text", None)

    should_generate = created or instance.embedding is None or text_changed

    if should_generate:
        instance._original_text = instance.get_ready_text()
        if app_settings.CELERY_ENABLED:
            try:
                from celery import Task
                from .tasks import generate_content_embedding_task

                model_path = f"{sender._meta.app_label}.{sender._meta.model_name}"
                celery_task: Task = generate_content_embedding_task  # type: ignore
                celery_task.delay(instance.id, model_path)
                return
            except (ImportError, ModuleNotFoundError):
                logger.warning(
                    "DNF: CELERY_ENABLED is True, but celery is not installed! Falling back to background threads."
                )
            except Exception as celery_err:
                logger.error(
                    f"Celery broker is down ({celery_err}), falling back to threads."
                )

        transaction.on_commit(
            lambda: threading.Thread(
                target=_run_synchronous_content_update,
                args=(sender, instance.id),
                daemon=True,
            ).start()
        )


def register_like_signal(
    *,
    like_target,
    mode: Literal["m2m", "model"],
    user_field_name: str | None = None,
    content_field_name: str | None = None,
):
    from functools import partial

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
        if action not in ("post_add", "post_remove"):
            return

        user_field_name = kwargs.get("user_field_name")
        content_field_name = kwargs.get("content_field_name")

        try:
            from django.contrib.auth import get_user_model

            User = get_user_model()

            if not user_field_name or not content_field_name:
                relation_fields = [f for f in sender._meta.fields if f.is_relation]

                user_fields = [
                    f
                    for f in relation_fields
                    if f.related_model == User
                    or (
                        isinstance(f.related_model, type)
                        and issubclass(f.related_model, User)
                    )
                ]

                if len(user_fields) != 1:
                    raise ValueError(
                        f"DNF: Model {sender.__name__} must have exactly one relation to User. "
                        f"Found {len(user_fields)}."
                    )

                content_fields = [
                    f for f in relation_fields if f.name != user_fields[0].name
                ]

                if len(content_fields) != 1:
                    raise ValueError(
                        f"DNF: Model {sender.__name__} must have exactly one content relation. "
                        f"Found {len(content_fields)}."
                    )

                user_field_name = user_fields[0].name
                content_field_name = content_fields[0].name

            if reverse:
                _trigger_embedding_update(
                    instance,
                    sender,
                    instance.id,
                    user_field_name,
                    content_field_name,
                )
            else:
                for user_object in User.objects.filter(pk__in=pk_set):
                    _trigger_embedding_update(
                        user_object,
                        sender,
                        user_object.id,  # type: ignore
                        user_field_name,
                        content_field_name,
                    )

        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"DNF Error (M2M signal): {e}")

    is_m2m = mode == "m2m"

    label = like_target._meta.label_lower

    if is_m2m:
        m2m_changed.connect(
            partial(
                user_like_changed_m2m,
                user_field_name=user_field_name,
                content_field_name=content_field_name,
            ),
            sender=like_target,
            dispatch_uid=f"dnf_m2m_{label}",
            weak=False,
        )
    else:
        if not user_field_name or not content_field_name:
            raise ValueError(
                "For custom like model specify user_field_name and content_field_name"
            )

        post_save.connect(
            user_like_changed_model,
            sender=like_target,
            dispatch_uid=f"dnf_model_{label}",
            weak=False,
        )


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
            logger.error(
                "Celery enabled, triggered 'except (ImportError, ModuleNotFoundError)'"
            )
        except Exception as celery_err:
            logger.error(
                f"Celery broker is down ({celery_err}), falling back to threads."
            )

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
        logger.exception("DNF [SIGNAL ERROR] Caught exception during content update:")
        raise
    finally:
        connection.close()


def _run_synchronous_user_update(
    user_model, user_id, sender_model, user_field_name, content_field_name
):
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
