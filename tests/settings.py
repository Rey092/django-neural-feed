import os

SECRET_KEY = "fake-secret-key-for-testing-purposes-only"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django_neural_feed",
]

USE_TZ = True
