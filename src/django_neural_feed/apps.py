from django.apps import AppConfig

class DjangoNeuralFeedConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_neural_feed"
    verbose_name = "Django Neural Feed"

    def ready(self):
        import django_neural_feed.signals