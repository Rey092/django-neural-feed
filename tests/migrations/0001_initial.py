# tests/migrations/0001_initial.py
from django.db import migrations, models
import django.db.models.deletion
import pgvector.django


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("django_neural_feed", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="TestPost",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False
                    ),
                ),
                ("title", models.CharField(max_length=255)),
                ("embedding", pgvector.django.VectorField(dimensions=384)),
            ],
        ),
        migrations.CreateModel(
            name="TestUserAction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False
                    ),
                ),
                ("action_type", models.CharField(max_length=50)),
                (
                    "post",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="tests.testpost"
                    ),
                ),
            ],
        ),
    ]
