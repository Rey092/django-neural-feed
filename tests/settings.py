import os

SECRET_KEY = "fake-secret-key-for-testing-purposes-only"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "django_neural_feed_test_db",
        "USER": "postgres",
        "PASSWORD": "password",
        "HOST": "localhost",
        "PORT": "5432",
        "TEST": {
            "NAME": "django_neural_feed_test_db",
        },
    }
}

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django_neural_feed",
    "tests",
]

AUTH_USER_MODEL = "tests.TestUser"

USE_TZ = True
