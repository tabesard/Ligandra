"""Data ingestion layer. Importing this package registers the built-in sources."""

# Import concrete sources so their @register decorators run.
from ligandra.data import chembl as _chembl  # noqa: F401,E402
from ligandra.data import local as _local  # noqa: F401,E402
from ligandra.data.base import DATA_SOURCES, DataSource

# BindingDB / PubChem / PDB are additional plugins; import lazily when present.
try:  # pragma: no cover - optional stubs
    from ligandra.data import bindingdb as _bindingdb  # noqa: F401
except Exception:
    pass

__all__ = ["DATA_SOURCES", "DataSource"]
