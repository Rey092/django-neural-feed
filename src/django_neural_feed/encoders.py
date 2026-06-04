import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseVectorEncoder:
    """
    Abstract interface for all vector encoders.
    Developers can subclass this to implement custom vector logic (e.g., OpenAI, Cohere).
    """

    @classmethod
    def text_to_vector(cls, text: str, model_name: str) -> list[float]:
        raise NotImplementedError("Subclasses must implement text_to_vector method.")

    @classmethod
    def average_vectors(cls, vectors: list[list[float]], limit: int) -> list[float]:
        """Default high-performance vector averaging using NumPy."""
        if not vectors:
            return []
        try:
            matrix = np.array(vectors[:limit], dtype=np.float32)
            return np.mean(matrix, axis=0).tolist()
        except Exception as e:
            logger.error(f"DNF Vector averaging failed: {e}")
            return []


class DefaultVectorEncoder(BaseVectorEncoder):
    """
    Built-in default encoder using the 'sentence-transformers' library.
    Generates embeddings locally without external API calls.
    """

    _model_instance = None

    @classmethod
    def _get_model(cls, model_name: str):
        """Lazy loading of the embedding model to save memory on app startup."""
        if cls._model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Model is downloaded automatically on the first execution
                cls._model_instance = SentenceTransformer(model_name)
            except ImportError:
                logger.error(
                    "DNF Error: 'sentence-transformers' package is missing. "
                    "Please install it via 'pip install sentence-transformers' "
                    "or configure a custom ENCODER_CLASS."
                )
                raise
        return cls._model_instance

    @classmethod
    def text_to_vector(cls, text: str, model_name: str) -> list[float]:
        if not text.strip():
            return []
        try:
            model = cls._get_model(model_name)
            # convert_to_numpy=True ensures we get a clean arrays back
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"DNF Default embedding generation failed: {e}")
            return []
