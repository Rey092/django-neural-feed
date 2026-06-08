# Contributing to django-neural-feed

First off, thank you for considering contributing to django-neural-feed! It's people like you that make django-neural-feed a great library for the community.

## 1. Branching Strategy

django-neural-feed uses short-lived feature and fix branches with `main` as the release branch:

- **`main`**: Production-ready code. Releases are cut from here.
- **`feature/*`**: Focused feature branches.
- **`fix/*`** or **`hotfix/*`**: Focused bug-fix branches.

Branch from `main` unless a maintainer asks you to target another branch.

## 2. Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/django-neural-feed.git
   cd django-neural-feed
   ```
2. Create your feature branch: `git checkout -b feature/your-feature-name`
3. Install the package with development tools:
   ```bash
   pip install -e ".[dev]"
   ```
4. Database for Testing:
This package relies heavily on PostgreSQL and the pgvector extension. The easiest way to run a local instance is via Docker:

```bash
docker run --name dnf-postgres -e POSTGRES_DB=django_neural_feed_test_db -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d pgvector/pgvector:pg16
```

## 3. Code Style

We enforce standard Python formatting:
- Use `black` for code formatting.
- Use `ruff` for linting.
- Use `pytest` for testing.

Before submitting a Pull Request, ensure your code passes all local checks and includes relevant test cases for new functionality.

Before submitting a PR, run:
```bash
black .
ruff check .
pytest --cov=src/django_neural_feed
```

## 4. Pull Request Process

1. Keep each PR focused on one bug, feature, or documentation improvement.
2. Ensure your code passes the relevant checks before pushing.
3. Update `README.md`, if you added a new feature.
4. Use the provided PR template and link the related issue with `Fixes #<issue-number>` when applicable.
5. Respond to maintainer feedback with a follow-up commit instead of opening a duplicate PR.
