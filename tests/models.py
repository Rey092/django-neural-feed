from django.db import models
from django.contrib.auth.models import AbstractUser
from django_neural_feed.mixins import NeuralRecommendMixin, NeuralUserMixin


class TestUser(NeuralUserMixin, AbstractUser):
    """Custom user model containing neural profile attributes for testing."""

    pass


class TestPost(NeuralRecommendMixin, models.Model):
    """Dummy post model for vector distance testing."""

    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)

    def __str__(self):
        return self.title

    def get_ready_text(self) -> str:
        return self.title


class TestUserAction(models.Model):
    """Dummy action model to store user interaction vectors."""

    id = models.AutoField(primary_key=True)

    user = models.ForeignKey(TestUser, on_delete=models.CASCADE)
    embedding = models.JSONField()
    action_type = models.CharField(max_length=10)  # 'like' or 'dislike'

    def __str__(self):
        return f"{self.user.username} - {self.action_type}"
