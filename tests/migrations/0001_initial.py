# tests/migrations/0001_initial.py
from django.db import migrations, models
import django.db.models.deletion
import pgvector.django
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        VectorExtension(),
        migrations.CreateModel(
            name="TestPost",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
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
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
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
