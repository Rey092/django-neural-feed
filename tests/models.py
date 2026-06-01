from django.db import models
from django.contrib.auth.models import AbstractUser
from django_neural_feed.mixins import NeuralRecommendMixin, NeuralUserMixin
from pgvector.django import VectorField


class TestUser(NeuralUserMixin, AbstractUser):
    pass


class TestPost(NeuralRecommendMixin, models.Model):
    title = models.CharField(max_length=255)

    def get_ready_text(self):
        return self.title


class TestM2MPost(NeuralRecommendMixin, models.Model):
    title = models.CharField(max_length=100)
    likes = models.ManyToManyField(related_name="liked_m2m_posts", to=TestUser)


class TestUserAction(models.Model):
    user = models.ForeignKey(TestUser, on_delete=models.CASCADE)
    post = models.ForeignKey(TestPost, on_delete=models.CASCADE)
    action_type = models.CharField(max_length=10)  # 'like'
