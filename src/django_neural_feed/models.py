from django.conf import settings
from django.db import models


class UserDislike(models.Model):
    """Negative reaction model. (aka 'not interested')"""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="neural_dislikes",
    )

    content_type = models.CharField(
        max_length=255, help_text="Model name"
    )
    object_id = models.PositiveIntegerField(help_text="Hidden object's id")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "content_type", "object_id"]),
        ]
        verbose_name = "Not interested content"
        verbose_name_plural = "Not interested content"

    def str(self):
        return f"User {self.user} -> {self.content_type} ({self.object_id})"