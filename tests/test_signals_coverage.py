# tests/test_signals_coverage.py
from functools import partial
import pytest
from unittest.mock import patch, MagicMock
from django.apps import apps
from django.db import connection

# Import signal handlers directly to bypass complex DB triggers
from django_neural_feed.signals import (
    generate_content_embedding,
    _user_like_changed_model,
    _user_like_changed_m2m,
    _trigger_user_embedding_update,
    _run_synchronous_content_update,
    _run_synchronous_user_update,
)
from django_neural_feed.mixins import NeuralRecommendMixin
from django_neural_feed.feeds import BaseNeuralFeed

# Note: Rename 'register_feed_signals' if your registration function is named differently
from django_neural_feed.signals import register_feed_signals

# --- 1. Line 17: Non-mixin sender guard ---


def test_generate_content_embedding_non_mixin_sender():
    """Covers line 17 early exit when sender lacks NeuralRecommendMixin."""

    class PlainModel:
        pass

    # Should return early without performing any actions
    assert (
        generate_content_embedding(sender=PlainModel, instance=None, created=True)
        is None
    )


# --- 2. Lines 28->exit & 32-39: Celery path and error fallbacks ---


def test_generate_content_embedding_celery_flow_and_errors():
    """Covers lines 28->exit and 32-39 for successful Celery dispatch and exceptions."""

    class MockMixinModel(NeuralRecommendMixin):
        id = 42
        embedding = None

        # Use MagicMock instead of a nested class to satisfy typing constraints
        _meta = MagicMock(app_label="tests", model_name="testarticle")

        def get_ready_text(self):
            return "Sample text"

    # Branch A: Celery is enabled and works smoothly
    with patch("django_neural_feed.signals.app_settings") as mock_settings:
        mock_settings.CELERY_ENABLED = True
        with patch(
            "django_neural_feed.tasks.generate_content_embedding_task"
        ) as mock_task:
            instance = MockMixinModel()
            generate_content_embedding(
                sender=MockMixinModel, instance=instance, created=True
            )
            mock_task.delay.assert_called_once_with(42, "tests.testarticle")

    # Branch B: Celery fails and triggers exception logger fallback
    with patch("django_neural_feed.signals.app_settings") as mock_settings:
        mock_settings.CELERY_ENABLED = True
        with patch(
            "django_neural_feed.tasks.generate_content_embedding_task"
        ) as mock_task:
            mock_task.delay.side_effect = Exception("Celery broker disconnected")
            instance = MockMixinModel()
            # Must catch the exception internally and proceed without crashing
            generate_content_embedding(
                sender=MockMixinModel, instance=instance, created=True
            )


# --- 3. Lines 55, 58, 65, 76: Signal registration routines ---


def test_signal_registration_logic_and_validation_errors():
    """Covers lines 55, 58, 65, and 76 during feed setup processing."""

    class DummyFeed(BaseNeuralFeed):
        pass

    # Line 75-76: Missing configuration fields raises ValueError
    with patch.object(DummyFeed, "get_setting", lambda attr: None):
        with pytest.raises(
            ValueError, match="Specify user_field_name and content_field_name"
        ):
            register_feed_signals(DummyFeed)

    # Line 55 & 58: Resolve string model name and exit if not found
    with patch.object(DummyFeed, "get_setting") as mock_get:
        mock_get.side_effect = lambda attr: (
            "invalid.ModelPath" if attr == "interaction_django_model" else "valid_field"
        )
        with patch.object(apps, "get_model", return_value=None):
            # Returns early on line 58 because like_target is evaluated as None
            assert register_feed_signals(DummyFeed) is None

    # Line 65: Successful connection setup for M2M mode interaction
    class ValidM2MFeed(BaseNeuralFeed):
        pass

    with patch.object(ValidM2MFeed, "get_setting") as mock_settings_get:
        mock_settings_get.side_effect = lambda attr: {
            "interaction_django_model": "tests.TestArticle",
            "interaction_mode": "m2m",
            "feed_id": "m2m_test",
            "user_field_name": "user",
            "content_field_name": "article",
        }.get(attr)

        with patch("django_neural_feed.signals.m2m_changed") as mock_m2m_signal:
            register_feed_signals(ValidM2MFeed)
            # Verify that connection loop targets the m2m pipeline completely
            assert mock_m2m_signal.connect.call_count > 0


# --- 4. Line 90 & 97-98: Model signal updates and exceptions ---


def test_user_like_changed_model_handlers():
    """Covers line 90 early exit and lines 97-98 exception logging block."""
    # Line 90: Return early if instance is updated rather than newly created
    assert (
        _user_like_changed_model(
            sender=None, instance=None, created=False, feed_class=None
        )
        is None
    )

    # Lines 97-98: Catch unexpected attribute lookup errors gracefully
    mock_feed = MagicMock()
    mock_feed.get_setting.side_effect = Exception("Dynamic configuration failure")
    _user_like_changed_model(
        sender=None, instance=MagicMock(), created=True, feed_class=mock_feed
    )


# --- 5. Lines 104-149: M2M processing pipeline ---


def test_user_like_changed_m2m_actions_and_flows():
    """Covers lines 104-149 processing logic for various M2M execution states."""
    # Line 104: Ignore non-relevant actions immediately
    assert (
        _user_like_changed_m2m(None, None, "pre_add", False, set(), feed_class=None)
        is None
    )

    # Run execution path for valid actions (post_add)
    mock_feed = MagicMock()
    mock_feed.get_setting.return_value = "user_property"
    mock_instance = MagicMock()
    setattr(mock_instance, "user_property", MagicMock(id=99))

    with patch(
        "django_neural_feed.signals._trigger_user_embedding_update"
    ) as mock_trigger:
        _user_like_changed_m2m(
            None, mock_instance, "post_add", False, {1, 2}, feed_class=mock_feed
        )
        mock_trigger.assert_called_once()


# --- 6. Lines 159-175: Celery user processing flows ---


class MockFeedClass(BaseNeuralFeed):
    pass


def test_trigger_user_embedding_update_celery_paths():
    """Covers lines 159-175 verifying user embedding task offloading."""
    # Branch A: Celery active route
    with patch("django_neural_feed.signals.app_settings") as mock_settings:
        mock_settings.CELERY_ENABLED = True
        with patch("django_neural_feed.tasks.update_user_embedding_task") as mock_task:
            _trigger_user_embedding_update(
                user_object=MagicMock(), sender=MagicMock(), feed_class=MockFeedClass
            )
            mock_task.delay.assert_called_once()

    # Branch B: Celery crash isolation route
    with patch("django_neural_feed.signals.app_settings") as mock_settings:
        mock_settings.CELERY_ENABLED = True
        with patch("django_neural_feed.tasks.update_user_embedding_task") as mock_task:
            mock_task.delay.side_effect = Exception("Celery task queue full")

            # Update both calls inside the test to use MockFeedClass instead of "test_feed"
            _trigger_user_embedding_update(
                user_object=MagicMock(),
                sender=MagicMock(),
                feed_class=MockFeedClass,  # Passed as class object
            )


# --- 7. Lines 195->206 & 202-204: Content fallback worker thread crashes ---


def test_run_synchronous_content_update_exception_handling():
    """Covers lines 195->206 and 202-204 verifying database or evaluation crashes."""
    mock_model = MagicMock()
    mock_model.objects.get.side_effect = Exception(
        "Database connection timeout during sync"
    )

    # Verifies that exception re-raises correctly while safely closing connection resources
    with pytest.raises(Exception, match="Database connection timeout"):
        _run_synchronous_content_update(model_class=mock_model, instance_id=1)


# --- 8. Line 229 & 232-239: User fallback worker thread guards and crashes ---


def test_run_synchronous_user_update_missing_embeddings_and_crashes():
    """Covers line 229 empty sequence exit and lines 232-239 thread exception safety."""
    # Line 229: Exit early when calculated vector collection returns empty results
    with patch.object(BaseNeuralFeed, "calculate_user_embedding", return_value=None):
        with patch("django_neural_feed.models.UserFeedProfile.objects.get_or_create"):
            assert (
                _run_synchronous_user_update(
                    user_id=1,
                    sender_model=MagicMock(),
                    feed_class=BaseNeuralFeed,
                    feed_id="test",
                )
                is None
            )

    # Lines 232-239: Intercept execution level crashes securely inside the background thread execution block
    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.get_or_create"
    ) as mock_db:
        mock_db.side_effect = Exception("Thread internal operational failure")
        # Function must intercept the error silently, log it safely, and release resources via finally block
        _run_synchronous_user_update(
            user_id=1,
            sender_model=MagicMock(),
            feed_class=BaseNeuralFeed,
            feed_id="test",
        )
