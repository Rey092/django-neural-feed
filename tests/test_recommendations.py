import pytest
import numpy as np
import time
from unittest.mock import PropertyMock
from django.contrib.auth import get_user_model
from django_neural_feed.services import RecommendationService
from tests.models import TestPost, TestUserAction, TestM2MPost
from django.db.models import Value
from django_neural_feed.signals import register_like_signal

User = get_user_model()

# ==============================================================================
# UNIT TESTS (MOCKED)
# ==============================================================================


def test_calculate_embedding_calls_sentence_transformers_correctly(mocker):
    """Ensure sentence-transformers initializes and encodes text correctly."""
    test_text = "Hello World!"
    fake_numpy_embedding = np.array([0.1, -0.5, 0.9, 0.0])
    expected_list = [0.1, -0.5, 0.9, 0.0]

    mock_model_instance = mocker.MagicMock()
    mock_model_instance.encode.return_value = fake_numpy_embedding

    mock_transformer_class = mocker.patch(
        "sentence_transformers.SentenceTransformer", return_value=mock_model_instance
    )

    RecommendationService._model_instance = None
    result = RecommendationService.calculate_embedding(test_text)

    mock_transformer_class.assert_called_once_with("intfloat/multilingual-e5-small")
    mock_model_instance.encode.assert_called_once_with(test_text, convert_to_numpy=True)

    assert isinstance(result, list)
    assert result == expected_list


@pytest.mark.django_db
def test_calculate_user_embedding_calculates_mean_correctly(mocker):
    """Validate mean vector math logic using pure deep mocking."""
    mocker.patch("sentence_transformers.SentenceTransformer")
    vector_1 = [1.0, 2.0, 3.0]
    vector_2 = [2.0, 4.0, 6.0]
    vector_3 = [3.0, 6.0, 9.0]
    expected_mean = [2.0, 4.0, 6.0]

    mock_queryset = mocker.MagicMock()
    mock_queryset.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        vector_1,
        vector_2,
        vector_3,
    ]

    result = RecommendationService.calculate_user_embedding(
        mock_queryset, content_field_name="post"
    )
    assert result == expected_mean


def test_get_ready_text_not_implemented_error():
    from django.db import models
    from django_neural_feed.mixins import NeuralRecommendMixin

    class BrokenDummyModel(NeuralRecommendMixin):
        pass

    instance = BrokenDummyModel()

    with pytest.raises(NotImplementedError):
        instance.get_ready_text()


@pytest.mark.django_db
def test_calculate_user_embedding_with_no_likes(mocker):
    from django_neural_feed.services import RecommendationService

    likes_qs = TestUserAction.objects.none()

    result = RecommendationService.calculate_user_embedding(
        likes_queryset=likes_qs, content_field_name="post"
    )

    assert result is None


def test_update_user_embedding_task_exception_handling(mocker, caplog):
    from django_neural_feed.tasks import update_user_embedding_task
    import logging

    mocker.patch(
        "django_neural_feed.tasks.get_model_from_path",
        side_effect=Exception("Database connection closed failure"),
    )

    with caplog.at_level(logging.ERROR):
        update_user_embedding_task(
            likes_model_path="tests.TestLike",
            users_model_path="tests.User",
            user_id=1,
            user_field_name="user",
            content_field_name="post",
        )

    assert len(caplog.records) == 1
    assert "Database connection closed failure" in caplog.text


@pytest.mark.django_db
def test_post_save_signal_no_trigger_if_text_unchanged(mocker):
    post = TestM2MPost.objects.create(title="Same text")
    mock_thread = mocker.patch("threading.Thread")

    post.save()
    mock_thread.assert_not_called()


def test_register_model_signal_requires_fields():
    with pytest.raises(ValueError):
        register_like_signal(TestM2MPost, mode="model")


@pytest.mark.django_db(transaction=True)
def test_post_save_celery_broker_down_fallback_to_thread(mocker):
    from django_neural_feed.conf import app_settings

    mocker.patch.object(
        type(app_settings),
        "CELERY_ENABLED",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )

    mocker.patch(
        "django_neural_feed.tasks.generate_content_embedding_task.delay",
        side_effect=Exception("Connection refused"),
    )
    mock_logger = mocker.patch("django_neural_feed.signals.logger")
    mock_thread = mocker.patch("threading.Thread")

    TestM2MPost.objects.create(title="Celery down post")

    mock_logger.error.assert_called_once()
    mock_thread.assert_called_once()


@pytest.mark.django_db
def test_run_synchronous_content_update_handles_exception(mocker):
    from django_neural_feed.signals import _run_synchronous_content_update

    mock_logger = mocker.patch("django_neural_feed.signals.logger")

    with pytest.raises(Exception):
        _run_synchronous_content_update(TestM2MPost, 99999)

    mock_logger.exception.assert_called_once()


@pytest.mark.django_db
def test_run_synchronous_user_update_handles_exception(caplog):
    import logging
    from django.contrib.auth import get_user_model
    from django_neural_feed.signals import _run_synchronous_user_update

    with caplog.at_level(logging.ERROR):
        _run_synchronous_user_update(
            get_user_model(), 99999, TestM2MPost, "user", "post"
        )

    assert any(
        "DNF Background Thread Error (User)" in record.message
        for record in caplog.records
    )


def test_user_like_changed_m2m_invalid_user_relation_logging(mocker, caplog):
    import logging
    from django.db.models.signals import m2m_changed

    mock_sender = mocker.MagicMock()
    mock_sender._meta.fields = []
    mock_sender.__name__ = "FakeM2MModel"
    mock_sender._meta.label_lower = "tests.fakem2mmodel"

    register_like_signal(mock_sender, mode="m2m")

    with caplog.at_level(logging.ERROR):
        m2m_changed.send(
            sender=mock_sender,
            instance=mocker.MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={1},
        )

    assert any(
        "must have exactly one relation to User" in record.message
        for record in caplog.records
    )


def test_user_like_changed_m2m_invalid_content_relation_logging(mocker, caplog):
    import logging
    from django.db.models.signals import m2m_changed
    from django.contrib.auth import get_user_model

    User = get_user_model()

    mock_user_field = mocker.MagicMock()
    mock_user_field.is_relation = True
    mock_user_field.related_model = User
    mock_user_field.name = "user"

    mock_sender = mocker.MagicMock()
    mock_sender._meta.fields = [mock_user_field]
    mock_sender.__name__ = "FakeM2MModel"
    mock_sender._meta.label_lower = "tests.fakem2mmodel"

    register_like_signal(mock_sender, mode="m2m")

    with caplog.at_level(logging.ERROR):
        m2m_changed.send(
            sender=mock_sender,
            instance=mocker.MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={1},
        )

    assert any(
        "must have exactly one content relation" in record.message
        for record in caplog.records
    )


@pytest.mark.django_db(transaction=True)
def test_m2m_signal_celery_not_installed_fallback(mocker):
    import sys
    from django_neural_feed.conf import app_settings
    from django.contrib.auth import get_user_model

    mocker.patch.object(
        type(app_settings),
        "CELERY_ENABLED",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )

    mocker.patch.dict(sys.modules, {"django_neural_feed.tasks": None})

    mock_thread = mocker.patch("threading.Thread")

    register_like_signal(TestM2MPost.likes.through, mode="m2m")

    User = get_user_model()
    user = User.objects.create(username="celery_missing_user")
    post = TestM2MPost.objects.create(title="Celery missing post")

    post.likes.add(user)

    mock_thread.assert_called_once()


# ==============================================================================
# INTEGRATION TESTS (REAL DATABASE & PGVECTOR)
# ==============================================================================


@pytest.mark.django_db
def test_calculate_user_embedding_with_real_db(mocker):
    """Verify mean vector aggregation using actual PostgreSQL records."""
    mocker.patch("sentence_transformers.SentenceTransformer")

    mocker.patch.object(
        RecommendationService, "calculate_embedding", return_value=[1.0, 2.0, 3.0]
    )

    user = User.objects.create_user(username="db_tester", password="password123")

    posts = TestPost.objects.bulk_create(
        [
            TestPost(title="P1", embedding=[1.0, 2.0, 3.0]),
            TestPost(title="P2", embedding=[2.0, 4.0, 6.0]),
            TestPost(title="P3", embedding=[3.0, 6.0, 9.0]),
        ]
    )

    TestUserAction.objects.create(user=user, post=posts[0], action_type="like")


@pytest.mark.django_db
def test_get_feed_for_user_without_user_embedding(mocker):
    """Verify that feed is working without user_embedding."""
    from django.db.models import Value
    from django_neural_feed.conf import app_settings
    from django_neural_feed.services import RecommendationService

    user = User.objects.create_user(username="feed_tester", password="password123")

    post1 = TestPost.objects.create(title="Post #1", embedding=[0.1, 0.2, 0.3])
    post2 = TestPost.objects.create(title="Post #2", embedding=[0.1, 0.2, 0.3])
    post3 = TestPost.objects.create(title="Post #3", embedding=[0.1, 0.2, 0.3])

    feed = RecommendationService.get_feed_for_user(
        user=user,
        model_class=TestPost,
        queryset=TestPost.objects.all(),
        likes_queryset=TestUserAction.objects.filter(user=user, action_type="like"),
        excluded_ids=[],
        limit=10,
    )
    feed = list(feed)
    assert len(feed) == 3
    assert feed[0].id > feed[1].id > feed[2].id


@pytest.mark.django_db
def test_get_feed_for_user_sorting_and_filtering(mocker):
    """Verify pgvector distance sorting and exclusion logic in database query."""
    from django.db.models import Value
    from django_neural_feed.conf import app_settings
    from django_neural_feed.services import RecommendationService

    mocker.patch.object(
        app_settings,
        "_user_config",
        {
            "WEIGHT_SIMILARITY": 1.0,
            "WEIGHT_FRESHNESS": 0.0,
            "WEIGHT_POPULARITY": 0.0,
            "POPULARITY_EXPRESSION": Value(0.0),
            "FRESHNESS_EXPRESSION": Value(0.0),
        },
    )

    mocker.patch("sentence_transformers.SentenceTransformer")

    user = User.objects.create_user(username="feed_tester", password="password123")

    posts = TestPost.objects.bulk_create(
        [
            TestPost(title="Close Match", embedding=[0.9, 0.1, 0.0]),
            TestPost(title="Far Match", embedding=[0.0, 0.1, 0.9]),
            TestPost(title="Disliked Item", embedding=[0.8, 0.0, 0.1]),
        ]
    )

    post_closest, post_far, post_disliked = posts

    user.user_embedding = [1.0, 0.0, 0.0]  # type: ignore
    user.save(update_fields=["user_embedding"])

    feed = RecommendationService.get_feed_for_user(
        user=user,
        model_class=TestPost,
        queryset=TestPost.objects.all(),
        likes_queryset=TestUserAction.objects.filter(user=user, action_type="like"),
        excluded_ids=[post_disliked.id],  # type: ignore
        limit=10,
    )

    assert feed.count() == 2
    assert feed[0].id == post_closest.id  # type: ignore
    assert feed[1].id == post_far.id  # type: ignore


class SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, *extra_args, **extra_kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        print(f"\n[SyncThread INIT] target: {target}, name: {extra_kwargs.get('name')}")

    def start(self):
        if self.target:
            print(f"[SyncThread START] Running {self.target.__name__} sync...")
            self.target(*self.args, **self.kwargs)


@pytest.fixture
def sync_like_signal_env(mocker):
    from django_neural_feed.conf import app_settings

    mocker.patch.dict(app_settings._user_config, {"CELERY_ENABLED": False})

    mocker.patch("django_neural_feed.signals.connection.close", lambda: None)
    mocker.patch("django_neural_feed.signals.transaction.on_commit", lambda f: f())
    mocker.patch("django_neural_feed.signals.threading.Thread", SyncThread)
    mocker.patch(
        "logging.Logger.error",
        side_effect=lambda msg, *args, **kwargs: pytest.fail(f"Logged error: {msg}"),
    )


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "add_relation",
    [
        pytest.param("forward", id="post.likes.add(user)"),
        pytest.param("reverse", id="user.liked_posts.add(post)"),
    ],
)
def test_m2m_like_signal_updates_user_embedding_bg_thread(
    mocker,
    sync_like_signal_env,
    add_relation,
):
    mocker.patch(
        "django_neural_feed.services.RecommendationService.calculate_embedding",
        return_value=[0.1, 0.2, 0.3],
    )
    mock_user_calc = mocker.patch(
        "django_neural_feed.services.RecommendationService.calculate_user_embedding",
        return_value=[0.5, -0.1, 0.8],
    )

    register_like_signal(TestM2MPost.likes.through, mode="m2m")

    user = User.objects.create(username="m2m_bg_user")
    post = TestM2MPost.objects.create(title="Thread testing django!")

    if add_relation == "forward":
        post.likes.add(user)
    else:
        user.liked_posts.add(post)  # type: ignore

    user.refresh_from_db()

    np.testing.assert_array_almost_equal(user.user_embedding, [0.5, -0.1, 0.8])  # type: ignore
    mock_user_calc.assert_called_once()


@pytest.mark.django_db
def test_post_save_signal_triggers_celery(mocker):
    from django_neural_feed.conf import app_settings

    mocker.patch.object(
        type(app_settings),
        "CELERY_ENABLED",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )

    mock_celery_delay = mocker.patch(
        "django_neural_feed.tasks.generate_content_embedding_task.delay"
    )

    post = TestM2MPost.objects.create(title="Celery post trigger")

    mock_celery_delay.assert_called_once_with(post.id, "tests.testm2mpost")  # type: ignore


@pytest.mark.django_db
def test_m2m_signal_triggers_celery(mocker):
    from django_neural_feed.conf import app_settings
    from django.contrib.auth import get_user_model

    mocker.patch.object(
        type(app_settings),
        "CELERY_ENABLED",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )

    mocker.patch("django_neural_feed.tasks.generate_content_embedding_task.delay")
    mock_user_celery_delay = mocker.patch(
        "django_neural_feed.tasks.update_user_embedding_task.delay"
    )

    register_like_signal(TestM2MPost.likes.through, mode="m2m")

    User = get_user_model()
    user = User.objects.create(username="celery_m2m_user")
    post = TestM2MPost.objects.create(title="Celery M2M post trigger")

    post.likes.add(user)

    mock_user_celery_delay.assert_called_once()


@pytest.mark.django_db
def test_generate_content_embedding_task_success(mocker):
    mocker.patch(
        "django_neural_feed.services.RecommendationService.calculate_embedding",
        return_value=[0.1, 0.2, 0.3],
    )
    from django_neural_feed.tasks import generate_content_embedding_task

    post = TestM2MPost.objects.create(title="Execute task content body")
    model_path = f"{post._meta.app_label}.{post._meta.model_name}"

    generate_content_embedding_task(post.id, model_path)  # type: ignore

    post.refresh_from_db()
    np.testing.assert_array_almost_equal(post.embedding, [0.1, 0.2, 0.3])


@pytest.mark.django_db
def test_update_user_embedding_task_success(mocker):
    mocker.patch(
        "django_neural_feed.services.RecommendationService.calculate_user_embedding",
        return_value=np.array([0.7, 0.8, 0.9]),
    )
    from django_neural_feed.tasks import update_user_embedding_task
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = User.objects.create(username="execute_task_user")
    post = TestM2MPost.objects.create(title="Execute task M2M body")

    register_like_signal(TestM2MPost.likes.through, mode="m2m")
    post.likes.add(user)

    through_model = TestM2MPost.likes.through
    likes_model_path = (
        f"{through_model._meta.app_label}.{through_model._meta.model_name}"
    )
    users_model_path = f"{user._meta.app_label}.{user._meta.model_name}"

    user_field_name = ""
    content_field_name = ""
    for field in through_model._meta.fields:
        if field.is_relation:
            if field.related_model == User:
                user_field_name = field.name
            elif field.related_model == TestM2MPost:
                content_field_name = field.name

    update_user_embedding_task(
        likes_model_path, users_model_path, user.id, user_field_name, content_field_name  # type: ignore
    )

    user.refresh_from_db()
    np.testing.assert_array_almost_equal(user.user_embedding, [0.7, 0.8, 0.9])  # type: ignore


def test_get_model_from_path_invalid_scenarios():
    from django_neural_feed.tasks import get_model_from_path

    assert get_model_from_path("invalidpath") is None
    assert get_model_from_path("non_existent_app.FakeModel") is None


@pytest.mark.django_db
def test_tasks_early_return_on_missing_model():
    from django_neural_feed.tasks import (
        generate_content_embedding_task,
        update_user_embedding_task,
    )

    assert generate_content_embedding_task(1, "bad.path") is None
    assert update_user_embedding_task("bad.path", "bad.path", 1, "u", "c") is None


@pytest.mark.django_db
def test_update_user_embedding_task_user_does_not_exist():
    from django_neural_feed.tasks import update_user_embedding_task
    from django.contrib.auth import get_user_model

    User = get_user_model()
    through_model = TestM2MPost.likes.through
    likes_model_path = (
        f"{through_model._meta.app_label}.{through_model._meta.model_name}"
    )
    users_model_path = f"{User._meta.app_label}.{User._meta.model_name}"

    result = update_user_embedding_task(
        likes_model_path, users_model_path, 999999, "testuser", "testm2mpost"
    )
    assert result is None


@pytest.mark.django_db
def test_tasks_generic_exception_handling(mocker):
    from django_neural_feed.tasks import generate_content_embedding_task

    mocker.patch(
        "django_neural_feed.tasks.get_model_from_path",
        side_effect=RuntimeError("Fatal dump"),
    )

    generate_content_embedding_task(1, "tests.testm2mpost")
