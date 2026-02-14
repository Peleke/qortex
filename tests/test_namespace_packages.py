"""Verify PEP 420 implicit namespace package mechanics.

If someone accidentally adds a qortex/__init__.py at the namespace root,
all sub-packages break. These tests catch that before 1800+ tests fail
mysteriously.
"""

from pathlib import Path


class TestNamespacePackageResolution:
    """NS-1: qortex is an implicit namespace package (PEP 420)."""

    def test_qortex_has_no_init_file(self):
        """The namespace root must NOT have __init__.py."""
        import qortex

        # Implicit namespace packages have no __file__
        assert not hasattr(qortex, "__file__") or qortex.__file__ is None

    def test_three_packages_resolve_to_separate_paths(self):
        """observe, ingest, and core live in different filesystem trees."""
        import qortex.ingest
        import qortex.observe

        import qortex.core

        obs_path = Path(qortex.observe.__file__).resolve()
        ing_path = Path(qortex.ingest.__file__).resolve()
        core_path = Path(qortex.core.__file__).resolve()

        # All three must be in different parent directories
        assert obs_path.parent != ing_path.parent
        assert obs_path.parent != core_path.parent
        assert ing_path.parent != core_path.parent

    def test_observe_resolves_to_packages_dir(self):
        import qortex.observe

        assert "packages/qortex-observe" in str(Path(qortex.observe.__file__).resolve())

    def test_ingest_resolves_to_packages_dir(self):
        import qortex.ingest

        assert "packages/qortex-ingest" in str(Path(qortex.ingest.__file__).resolve())

    def test_core_resolves_to_src_dir(self):
        import qortex.core

        assert "src/qortex/core" in str(Path(qortex.core.__file__).resolve())

    def test_no_namespace_init_py_on_disk(self):
        """Guard: no qortex/__init__.py in any of the three source trees."""
        import qortex.ingest
        import qortex.observe

        import qortex.core

        for mod in (qortex.observe, qortex.ingest, qortex.core):
            pkg_dir = Path(mod.__file__).resolve().parent  # e.g. .../qortex/observe
            namespace_dir = pkg_dir.parent  # e.g. .../qortex/
            init_file = namespace_dir / "__init__.py"
            assert not init_file.exists(), (
                f"Found {init_file} -- this will break PEP 420 namespace packages!"
            )

    def test_cross_namespace_import(self):
        """qortex.ingest can import from qortex.core (cross-package)."""
        from qortex.ingest import IngestionManifest

        from qortex.core.models import IngestionManifest as CoreManifest

        assert IngestionManifest is CoreManifest
