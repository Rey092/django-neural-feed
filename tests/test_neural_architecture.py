import pytest
from django.contrib.auth import get_user_model
from django.db import transaction
from tests.models import TestArticle, TestLikeModel
from tests.feeds import TestParentFeed, TestChildFeed
from django_neural_feed.models import UserFeedProfile
import numpy as np
import time

from django_neural_feed.tasks import (
    generate_content_embedding_task,
    update_user_embedding_task,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_content_embedding_trigger():
    """Checks that creating content triggers embedding generation."""

    with transaction.atomic():
        article = TestArticle.objects.create(title="Python Django Deep Learning")

    for _ in range(10):
        article.refresh_from_db()
        if article.embedding is not None:
            break
        time.sleep(0.1)

    assert article.embedding is not None
    assert len(article.embedding) == 3


@pytest.mark.django_db
def test_user_profile_and_parent_inheritance():
    """Tests that user interaction updates profile and child feed inherits it."""
    from django.contrib.auth import get_user_model
    from django_neural_feed.models import UserFeedProfile
    from tests.feeds import TestChildFeed
    import numpy as np

    User = get_user_model()
    user = User.objects.create_user(username="testuser", password="password")

    article1 = TestArticle.objects.create(title="A")
    article2 = TestArticle.objects.create(title="B")

    TestLikeModel.objects.create(user=user, article=article1)
    TestLikeModel.objects.create(user=user, article=article2)

    profile = UserFeedProfile.objects.filter(user=user, feed_id="test_parent").first()

    assert profile is not None
    assert profile.embedding is not None

    child_vector = TestChildFeed.get_user_vector(user)
    assert child_vector is not None
    assert np.allclose(child_vector, profile.embedding)


@pytest.mark.django_db
def test_content_embedding_update_on_text_change():
    """Checks that modifying text triggers embedding regeneration."""
    article = TestArticle.objects.create(title="Initial Title")
    article.refresh_from_db()
    assert np.allclose(article.embedding, [0.5, 0.5, 0.5])

    article.title = "Completely New Title"
    article.save()

    article.refresh_from_db()
    assert article.embedding is not None


@pytest.mark.django_db
def test_celery_tasks_execution(settings, monkeypatch):
    """Tests that Celery tasks run successfully with accurate app model paths."""
    from django_neural_feed.conf import app_settings

    settings.DJANGO_NEURAL_FEED = {
        **getattr(settings, "DJANGO_NEURAL_FEED", {}),
        "CELERY_ENABLED": True,
        "VECTOR_DIMENSION": 3,
    }

    try:
        from celery import current_app

        current_app.conf.task_always_eager = True  # type: ignore
    except ImportError:
        pass

    article = TestArticle.objects.create(title="Celery Test Article")

    generate_content_embedding_task(
        django_model_path="tests.testarticle", instance_id=article.id  # type: ignore
    )

    article.refresh_from_db()
    assert np.allclose(article.embedding, [0.5, 0.5, 0.5])

    user = User.objects.create_user(username="celeryuser", password="password")
    TestLikeModel.objects.create(user=user, article=article)

    update_user_embedding_task(
        likes_django_model_path="tests.TestLikeModel",
        users_django_model_path="auth.User",
        user_id=user.id,  # type: ignore
        user_field_name="user",
        content_field_name="article",
        feed_id="test_parent",
        user_likes_limit=3,
    )

    from django_neural_feed.models import UserFeedProfile

    profile = UserFeedProfile.objects.filter(user_id=user.id, feed_id="test_parent").first()  # type: ignore
    assert profile is not None
    assert np.allclose(profile.embedding, [0.57735026, 0.57735026, 0.57735026])
