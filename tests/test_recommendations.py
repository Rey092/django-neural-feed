import pytest
import numpy as np
from django_neural_feed.services import RecommendationService


def test_calculate_embedding_calls_sentence_transformers_correctly(mocker):
    test_text = "Hello World!"

    fake_numpy_embedding = np.array([0.1, -0.5, 0.9, 0.0])
    expected_list = [0.1, -0.5, 0.9, 0.0]

    mock_model_instance = mocker.MagicMock()
    mock_model_instance.encode.return_value = fake_numpy_embedding

    mock_transformer_class = mocker.patch(
        "sentence_transformers.SentenceTransformer", return_value=mock_model_instance
    )

    RecommendationService._model_instance = None

    result = RecommendationService.calculate_embedding(test_text)

    mock_transformer_class.assert_called_once_with("intfloat/multilingual-e5-small")

    mock_model_instance.encode.assert_called_once_with(test_text, convert_to_numpy=True)

    assert isinstance(result, list)
    assert result == expected_list


def test_calculate_user_embedding_calculates_mean_correctly(mocker):
    vector_1 = [1.0, 2.0, 3.0]
    vector_2 = [2.0, 4.0, 6.0]
    vector_3 = [3.0, 6.0, 9.0]

    expected_mean = [2.0, 4.0, 6.0]

    mock_queryset = mocker.MagicMock()

    mock_queryset.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        vector_1,
        vector_2,
        vector_3,
    ]

    result = RecommendationService.calculate_user_embedding(mock_queryset)
    assert result == expected_mean
