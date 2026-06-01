import pytest
import numpy as np
from django.contrib.auth import get_user_model
from django_neural_feed.services import RecommendationService
from tests.models import TestPost, TestUserAction

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


def test_calculate_user_embedding_calculates_mean_correctly(mocker):
    """Validate mean vector math logic using pure deep mocking."""
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

    result = RecommendationService.calculate_user_embedding(mock_queryset)
    assert result == expected_mean


# ==============================================================================
# INTEGRATION TESTS (REAL DATABASE & PGVECTOR)
# ==============================================================================


@pytest.mark.django_db
def test_calculate_user_embedding_with_real_db():
    """Verify mean vector aggregation using actual PostgreSQL records."""
    user = User.objects.create_user(username="db_tester", password="password123")

    # Save real rows with distinct embeddings
    TestUserAction.objects.create(
        user=user, embedding=[1.0, 2.0, 3.0], action_type="like"
    )
    TestUserAction.objects.create(
        user=user, embedding=[2.0, 4.0, 6.0], action_type="like"
    )
    TestUserAction.objects.create(
        user=user, embedding=[3.0, 6.0, 9.0], action_type="like"
    )

    queryset = TestUserAction.objects.filter(user=user, action_type="like")
    result = RecommendationService.calculate_user_embedding(queryset)

    assert result == [2.0, 4.0, 6.0]


@pytest.mark.django_db
def test_get_feed_for_user_sorting_and_filtering(mocker):
    """Verify pgvector distance sorting and exclusion logic in database query."""
    user = User.objects.create_user(username="feed_tester", password="password123")

    # Create target posts with specific vector coordinates
    post_closest = TestPost.objects.create(
        title="Close Match", embedding=[0.9, 0.1, 0.0]
    )
    post_far = TestPost.objects.create(title="Far Match", embedding=[0.0, 0.1, 0.9])
    post_disliked = TestPost.objects.create(
        title="Disliked Item", embedding=[0.8, 0.0, 0.1]
    )

    # Mock user profile vector to point towards [1.0, 0.0, 0.0]
    mocker.patch.object(
        RecommendationService, "calculate_user_embedding", return_value=[1.0, 0.0, 0.0]
    )

    # Execute main feed generation service
    feed = RecommendationService.get_feed_for_user(
        user=user,
        model_class=TestPost,
        queryset=TestPost.objects.all(),
        likes_queryset=TestUserAction.objects.filter(user=user, action_type="like"),
        excluded_ids=[post_disliked.id],  # type: ignore
        limit=10,
    )

    # Verify exclusions and pgvector cosine distance sorting order
    assert feed.count() == 2
    assert (
        feed[0].id == post_closest.id
    )  # Must be first due to highest vector proximity # type: ignore
    assert feed[1].id == post_far.id  # type: ignore
