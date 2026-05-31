# **Django Neural Feed (DNF)**

Django Neural Feed (DNF) is a Django application designed to build personalized content feeds using vector embeddings. It utilizes PostgreSQL's pgvector extension for semantic ranking and combines it with content popularity and freshness metrics directly inside the database query.

## **Features**

* **Custom Model Support:** Connects to your existing Like, Dislike, or Hide models using Django signals. No database migrations are required for your existing models.  
* **Optional Async Processing:** Offloads heavy embedding generation and user profile updates to background tasks via Celery.  
* **Hybrid Scoring:** Combines semantic vector similarity (Cosine Distance) with content freshness and popularity weights in a single database query.  
* **Built-in Filtering:** Excludes disliked, hidden, or viewed content directly inside the SQL query using lazy evaluation.

## **Installation**

1. Install the package via pip:
```bash
pip install django-neural-feed
```
2. Add django_neural_feed to your INSTALLED_APPS in settings.py:
```python
INSTALLED_APPS = [  
    ...,  
    'django_neural_feed',  
]
```
## **Quick Start**

### **1. Update Your Models**

Add the mixins to your target content model (e.g., Post) and your User model:
```python
from django.db import models  
from django.contrib.auth.models import AbstractUser  
from django_neural_feed.mixins import NeuralRecommendMixin, NeuralUserMixin

class CustomUser(AbstractUser, NeuralUserMixin):  
    pass

class Post(models.Model, NeuralRecommendMixin):  
    title = models.CharField(max_length=255)  
    content = models.TextField()  
    likes_count = models.PositiveIntegerField(default=0)  
    created_at = models.DateTimeField(auto_now_add=True)

    # Return the text that should be used to generate the vector embedding  
    def get_ready_text(self) -> str:  
        return f"{self.title} {self.content}"
```
### **2. Register Your Interaction Model**

Connect your custom Like/Interaction model in your app's apps.py file:
```python
from django.apps import AppConfig

class YourAppConfig(AppConfig):  
    name = 'your_app'

    def ready(self):  
        from django_neural_feed.signals import register_like_signal  
        from .models import Like

        # Automatically updates user preference embeddings when a new Like is created  
        register_like_signal(  
            like_model_class=Like,   
            user_field_name='user',    # Field pointing to User model  
            content_field_name='post'  # Field pointing to Content model  
        )
```
### **3. Get the Feed**

Pass your querysets to the RecommendationService to get a ranked and filtered feed:
```python
from django_neural_feed.services import RecommendationService  
from .models import Post, Like

def my_feed_view(request):  
    # Get IDs of items to exclude (e.g., dislikes or hidden posts)  
    excluded_ids = Like.objects.filter(  
        user=request.user,   
        is_dislike=True  
    ).values_list('post_id', flat=True)  
      
    # Get user's active likes to calculate interests  
    user_likes = Like.objects.filter(user=request.user, is_dislike=False)

    feed_queryset = RecommendationService.get_feed_for_user(  
        user=request.user,  
        model_class=Post,  
        queryset=Post.objects.all(),  
        likes_queryset=user_likes,  
        excluded_ids=excluded_ids,  
        limit=20  
    )  
      
    return feed_queryset
```
## **Configuration Settings**

Add DNF_CONFIG to your settings.py to change default behaviors:
```python
DNF_CONFIG = {  
    "CELERY_ENABLED": True,  
    "USER_LIKES_LIMIT": 30,  
    "MODEL_NAME": "intfloat/multilingual-e5-small",  
    "WEIGHT_SIMILARITY": 0.6,  
    "WEIGHT_FRESHNESS": 0.2,  
    "WEIGHT_POPULARITY": 0.2,  
}
```
### **Parameters**

| Parameter | Default Value | Description |
| :---- | :---- | :---- |
| `CELERY_ENABLED` | `False` | Set to `True` to process embedding generation via Celery tasks in the background. |
| `USER_LIKES_LIMIT` | `20` | How many recent user likes are analyzed to calculate the average user interest profile vector. |
| `MODEL_NAME` | `'intfloat/multilingual-e5-small'` | The SentenceTransformer model used for text vectorization. |
| `WEIGHT_SIMILARITY` | `0.6` | Scoring multiplier for semantic vector similarity. |
| `WEIGHT_FRESHNESS` | `0.2` | Scoring multiplier for content freshness. |
| `WEIGHT_POPULARITY` | `0.2` | Scoring multiplier for content popularity. |
| `POPULARITY_EXPRESSION` | `Value(1.0)` | Django F-expression or database function used to get popularity values. Defaults to a neutral constant. |
| `FRESHNESS_EXPRESSION` | `Value(1.0)` | SQL expression or Django expression for the time-decay factor. Defaults to a neutral constant. |

## **License**

MIT
