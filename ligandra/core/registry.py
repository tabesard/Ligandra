"""Generic string-keyed registry used to make every layer pluggable.

The registry pattern is the backbone of Ligandra's extensibility guarantee:
adding a new data source, featurizer, model, generator or scorer means writing
one subclass and decorating it with ``@SOME_REGISTRY.register("name")``.  No
orchestration code changes.  See :mod:`ligandra.core.registry` usages across
the package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A minimal, typed name -> class registry.

    Parameters
    ----------
    kind:
        Human-readable label for the kind of thing being registered (used in
        error messages), e.g. ``"model"`` or ``"featurizer"``.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator that registers ``cls`` under ``name``.

        Raises
        ------
        ValueError
            If ``name`` is already registered (prevents silent shadowing).
        """

        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(
                    f"{self.kind} '{name}' is already registered "
                    f"({self._items[name].__name__})."
                )
            self._items[name] = cls
            # Stash the registered name on the class for round-tripping configs.
            cls.registry_name = name
            return cls

        return deco

    def create(self, name: str, **kwargs: object) -> T:
        """Instantiate the registered class ``name`` with ``kwargs``."""
        return self.get(name)(**kwargs)  # type: ignore[call-arg]

    def get(self, name: str) -> type[T]:
        """Return the registered class ``name`` (not an instance)."""
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. "
                f"Available: {', '.join(self.available()) or '(none)'}."
            ) from None

    def available(self) -> list[str]:
        """Sorted list of registered names (drives UI dropdowns)."""
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
