"""Registry pattern for view synthesis backends.

Provides dynamic registration and instantiation of view synthesis backends.
New backends can be added by simply decorating the class with @register.

Example:
    @ViewSynthesizerRegistry.register("my_backend")
    class MySynthesizer(ViewSynthesizerBase):
        ...

    # Later, instantiate by name
    synthesizer = ViewSynthesizerRegistry.create("my_backend", config)
"""

from typing import Dict, Type, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class ViewSynthesizerRegistry:
    """Registry for view synthesis backend implementations.

    Supports:
    - Dynamic registration via decorator
    - Factory method for creating instances by name
    - Discovery of available backends
    """

    _backends: Dict[str, Type] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, priority: int = 100, **metadata):
        """Decorator to register a view synthesis backend.

        Args:
            name: Unique identifier for the backend
            priority: Lower = higher priority for fallback selection
            **metadata: Additional metadata (description, requirements, etc.)

        Returns:
            Decorator function

        Example:
            @ViewSynthesizerRegistry.register("zeronvs", priority=50)
            class ZeroNVSSynthesizer(ViewSynthesizerBase):
                ...
        """
        def decorator(backend_cls: Type) -> Type:
            if name in cls._backends:
                logger.warning(
                    f"Backend '{name}' already registered. Overwriting."
                )

            cls._backends[name] = backend_cls
            cls._metadata[name] = {
                "priority": priority,
                "class": backend_cls.__name__,
                **metadata,
            }

            # Ensure backend has backend_name attribute
            if not hasattr(backend_cls, 'backend_name'):
                backend_cls.backend_name = name

            logger.debug(f"Registered view synthesis backend: {name}")
            return backend_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a registered backend class by name.

        Args:
            name: Backend identifier

        Returns:
            Backend class or None if not found
        """
        return cls._backends.get(name)

    @classmethod
    def create(cls, name: str, config: Any) -> Any:
        """Create an instance of a registered backend.

        Args:
            name: Backend identifier
            config: Configuration object to pass to constructor

        Returns:
            Instantiated backend

        Raises:
            KeyError: If backend not registered
        """
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise KeyError(
                f"Backend '{name}' not registered. "
                f"Available backends: {available}"
            )

        backend_cls = cls._backends[name]
        return backend_cls(config)

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names."""
        return list(cls._backends.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a registered backend."""
        return cls._metadata.get(name, {})

    @classmethod
    def get_by_priority(cls) -> List[str]:
        """Get backend names sorted by priority (lower = higher priority)."""
        return sorted(
            cls._backends.keys(),
            key=lambda n: cls._metadata.get(n, {}).get("priority", 100)
        )

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a backend is registered and can be imported."""
        if name not in cls._backends:
            return False

        # Try to instantiate with minimal config to check dependencies
        try:
            backend_cls = cls._backends[name]
            # Just check the class exists and is importable
            return True
        except Exception:
            return False


def create_view_synthesizer(config: Any) -> Any:
    """Factory function to create a view synthesizer from configuration.

    This is the main entry point for creating synthesizers. It handles:
    - Backend selection based on config
    - Fallback to default if specified backend unavailable
    - Configuration translation

    Args:
        config: Configuration object with 'backend' attribute or dict with 'backend' key.
               Supports ViewSynthesisConfig, dict, or any object with backend attribute.

    Returns:
        Instantiated view synthesizer

    Example:
        config = {"backend": "depth_warping", "rotation_angles": [-5, 0, 5]}
        synthesizer = create_view_synthesizer(config)
    """
    # Extract backend name from config
    if hasattr(config, 'backend'):
        backend_name = config.backend
    elif isinstance(config, dict):
        backend_name = config.get('backend', 'depth_warping')
    else:
        backend_name = 'depth_warping'  # Default

    # Handle hybrid/fallback mode
    if backend_name == "hybrid" or backend_name == "auto":
        # Import here to avoid circular imports
        from .hybrid import HybridSynthesizer
        return HybridSynthesizer(config)

    # Try to create the requested backend
    try:
        return ViewSynthesizerRegistry.create(backend_name, config)
    except KeyError:
        logger.warning(
            f"Backend '{backend_name}' not available, falling back to 'depth_warping'"
        )
        return ViewSynthesizerRegistry.create('depth_warping', config)


def ensure_backends_loaded():
    """Ensure all built-in backends are loaded and registered.

    Call this at module initialization to register all backends.
    """
    # Depth warping is always available
    from . import depth_warping  # noqa: F401

    # Try to load optional backends
    try:
        from . import zeronvs  # noqa: F401
    except ImportError:
        logger.debug("ZeroNVS backend not available (missing dependencies)")

    try:
        from . import zero123  # noqa: F401
    except ImportError:
        logger.debug("Zero123++ backend not available (missing dependencies)")

    try:
        from . import hybrid  # noqa: F401
    except ImportError:
        logger.debug("Hybrid backend not available")

    try:
        from . import mvgenmaster  # noqa: F401
    except ImportError:
        logger.debug("MVGenMaster backend not available (missing dependencies)")
