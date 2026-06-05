import pytest
from unittest.mock import MagicMock
from django.contrib.auth import get_user_model
from django.db import OperationalError
from django.db.models import Value
from django_neural_feed.feeds import BaseNeuralFeed
from django_neural_feed.models import UserFeedProfile
from unittest.mock import MagicMock, patch
from unittest.mock import patch, PropertyMock, MagicMock
from django_neural_feed.conf import AppSettings

# --- 1. Line 41: Fallback via getattr ---


def test_real_base_feed_get_setting_fallback():
    """Covers line 41 by requesting an attribute inherited from object."""
    # __sizeof__ is missing from BaseNeuralFeed.__dict__, forcing line 41
    assert BaseNeuralFeed.get_setting("__sizeof__") is not None


# --- 2. Lines 45-49: get_candidates with exclusions ---


@pytest.mark.django_db
def test_get_candidates_with_and_without_exclusions():
    """Covers lines 45-49 using a real database model."""
    from tests.models import TestArticle

    class TargetFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            return super().get_setting(attr_name)

    a1 = TestArticle.objects.create(title="A1", embedding=[1, 0, 0])
    a2 = TestArticle.objects.create(title="A2", embedding=[0, 1, 0])

    # Covers False branch transition
    qs_all = TargetFeed.get_candidates(user=None, queryset=None, excluded_ids=None)
    assert qs_all.count() == 2

    # Covers True branch filtering
    qs_excluded = TargetFeed.get_candidates(user=None, queryset=TestArticle.objects.all(), excluded_ids=[a1.id])  # type: ignore
    assert qs_excluded.count() == 1


# --- 3. Lines 55-77 & 68: calculate_user_embedding Math & Guards ---


def test_calculate_user_embedding_logic():
    """Covers lines 55-77, handling norm > 0, norm == 0, and empty states."""

    class MathTestingFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            return 5

    # Branch: norm > 0
    mock_qs = MagicMock()
    mock_qs.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [3.0, 0.0, 4.0]
    ]
    assert pytest.approx(MathTestingFeed.calculate_user_embedding(mock_qs)) == [
        0.6,
        0.0,
        0.8,
    ]

    # Branch: norm == 0
    mock_qs_zero = MagicMock()
    mock_qs_zero.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.0, 0.0, 0.0]
    ]
    assert MathTestingFeed.calculate_user_embedding(mock_qs_zero) == [0.0, 0.0, 0.0]

    # Line 68: Empty sequence early exit
    mock_qs_empty = MagicMock()
    mock_qs_empty.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = (
        []
    )
    assert MathTestingFeed.calculate_user_embedding(mock_qs_empty) is None


# --- 4. Lines 93 & 102-109: Guards and DB Exceptions ---


def test_get_user_profile_vector_guards():
    """Covers line 93 when feed_id is missing."""

    class BrokenFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            return None

    assert BrokenFeed.get_user_vector(user=MagicMock()) is None


@patch("django_neural_feed.models.UserFeedProfile.objects.filter")
def test_get_user_profile_vector_db_crash(mock_filter):
    """Covers lines 102-109 exception block."""

    class CrashingFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            return "crash_feed"

    mock_filter.side_effect = OperationalError("Database disconnected")
    assert CrashingFeed.get_user_vector(user=MagicMock()) is None


# --- 5. Lines 149-151: Real pgvector Slicing & Integrity ---


@pytest.mark.django_db
def test_generate_feed_slicing_flow():
    """Covers lines 149-151 down to the slice limit using real DB relations."""
    from tests.models import TestArticle

    class RealPipelineFeed(BaseNeuralFeed):
        popularity_expression = Value(0.0)
        freshness_expression = Value(0.0)
        weight_similarity = 1.0
        weight_freshness = 0.0
        weight_popularity = 0.0

        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            if attr_name == "feed_id":
                return "slice_feed"
            if "limit" in attr_name:
                return 1
            return super().get_setting(attr_name)

    # Create a persistent user to satisfy PostgreSQL foreign keys
    User = get_user_model()
    real_user = User.objects.create_user(username="real_test_user")

    UserFeedProfile.objects.create(user_id=real_user.id, feed_id="slice_feed", embedding=[1.0, 0.0, 0.0])  # type: ignore

    TestArticle.objects.create(title="First", embedding=[1.0, 0.0, 0.0])
    TestArticle.objects.create(title="Second", embedding=[0.9, 0.0, 0.0])

    # Pass limit explicitly to force slicing execution
    final_feed = RealPipelineFeed.get_feed(user=real_user, limit=1)
    assert len(final_feed) == 1


def test_calculate_embedding_resolves_encoder_and_model():
    """Covers calculate_embedding by overriding the ENCODER_CLASS property via class level patch."""

    class CustomModelFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "embedding_model_name":
                return "feed-specific-bert-model"
            return super().get_setting(attr_name)

    mock_encoder = MagicMock()
    mock_encoder.text_to_vector.return_value = [0.25, 0.5, 0.75]

    # Patch the property on the class level using PropertyMock
    with patch.object(
        AppSettings, "ENCODER_CLASS", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_encoder

        result = CustomModelFeed.calculate_embedding("Some raw content text")

        # Verify interactions
        mock_encoder.text_to_vector.assert_called_once_with(
            "Some raw content text", "feed-specific-bert-model"
        )
        assert result == [0.25, 0.5, 0.75]
