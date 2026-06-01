import pytest
import numpy as np
import time
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
    TestUserAction.objects.create(user=user, post=posts[1], action_type="like")
    TestUserAction.objects.create(user=user, post=posts[2], action_type="like")

    queryset = TestUserAction.objects.filter(user=user, action_type="like")
    result = RecommendationService.calculate_user_embedding(
        queryset, content_field_name="post"
    )

    assert result == [2.0, 4.0, 6.0]


@pytest.mark.django_db
def test_get_feed_for_user_sorting_and_filtering(mocker):
    """Verify pgvector distance sorting and exclusion logic in database query."""
    from django.db.models import Value
    from django_neural_feed.conf import app_settings
    from django_neural_feed.services import RecommendationService

    # Replace the PropertyMock mess by directly patching the instance's config dict
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

    print("DEBUG. User embedding is", user.user_embedding)  # type: ignore

    feed = RecommendationService.get_feed_for_user(
        user=user,
        model_class=TestPost,
        queryset=TestPost.objects.all(),
        likes_queryset=TestUserAction.objects.filter(user=user, action_type="like"),
        excluded_ids=[post_disliked.id],  # type: ignore
        limit=10,
    )

    for p in feed:
        print(
            p.title,
            p.similarity,
            p.score,
        )

    assert feed.count() == 2
    assert feed[0].id == post_closest.id  # type: ignore
    assert feed[1].id == post_far.id  # type: ignore


@pytest.mark.django_db(transaction=True)
def test_m2m_like_signal_updates_user_embedding_bg_thread(mocker):
    mock_calculate = mocker.patch(
        "django_neural_feed.services.RecommendationService.calculate_user_embedding",
        return_value=[0.5, -0.1, 0.8]
    )

    register_like_signal(TestM2MPost.likes.through)

    user = User.objects.create(username="m2m_bg_user")
    post = TestM2MPost.objects.create(title="Thread testing django!")

    post.embedding = [0.5, -0.1, 0.8]
    post.save()

    assert user.user_embedding is None  # type: ignore

    post.likes.add(user)

    updated_user = None
    for _ in range(20):
        user.refresh_from_db()
        if user.user_embedding is not None:  # type: ignore
            updated_user = user
            break
        time.sleep(0.02)

    assert updated_user is not None
    assert len(updated_user.user_embedding) == 3  # type: ignore
    mock_calculate.assert_called_once()