from django.db import models
from django.conf import settings
from pgvector.django import VectorField
from django_neural_feed.conf import app_settings


class UserFeedProfile(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="feed_profiles"
    )

    feed_id = models.CharField(max_length=64, db_index=True)

    embedding = VectorField(
        null=True,
        blank=True,
    )

    class Meta:
        unique_together = ("user", "feed_id")

    def __str__(self):
        return f"{self.user.username} - {self.feed_id}"
