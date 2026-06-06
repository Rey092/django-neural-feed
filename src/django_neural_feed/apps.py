from django.apps import AppConfig


class DjangoNeuralFeedConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_neural_feed"
    verbose_name = "Django Neural Feed"

    def ready(self):
        import django_neural_feed.signals
        from django_neural_feed.conf import app_settings
        from django_neural_feed.signals import (
            register_feed_signals,
            register_content_signals,
        )

        for feed_class in app_settings.get_registered_feeds():
            register_feed_signals(feed_class)

            content_django_model = feed_class.content_django_model
            if content_django_model:
                if isinstance(content_django_model, str):
                    from django.apps import apps

                    content_django_model = apps.get_model(content_django_model)

                register_content_signals(content_django_model)
