from django_neural_feed.feeds import BaseNeuralFeed
from .models import TestArticle, TestLikeModel


class TestParentFeed(BaseNeuralFeed):
    feed_id = "test_parent"
    content_django_model = TestArticle
    interaction_django_model = TestLikeModel
    user_field_name = "user"
    content_field_name = "article"
    user_likes_limit = 3


class TestChildFeed(BaseNeuralFeed):
    parent_feed = TestParentFeed
