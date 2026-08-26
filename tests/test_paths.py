from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from kilix_ollama.errors import RuntimeContractError
from kilix_ollama.paths import RuntimePaths, private_directory_status


class RuntimePathTests(unittest.TestCase):
    def test_layout_is_deterministic_and_does_not_create_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths.from_environment(
                {
                    "HOME": str(root / "home"),
                    "XDG_RUNTIME_DIR": str(root / "run"),
                    "XDG_STATE_HOME": str(root / "state"),
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                },
                uid=1234,
            )
            self.assertEqual(paths.mediation_socket, root / "run/kilix-ollama/api.sock")
            self.assertEqual(paths.model_store, root / "data/kilix-ollama/models")
            self.assertFalse(paths.runtime_dir.exists())

    def test_relative_xdg_root_is_rejected(self) -> None:
        with self.assertRaises(RuntimeContractError):
            RuntimePaths.from_environment(
                {"HOME": "/srv/test-user", "XDG_RUNTIME_DIR": "relative"},
                uid=1234,
            )

    def test_root_traversal_control_characters_and_invalid_uid_are_rejected(self) -> None:
        invalid_environments = (
            {"HOME": "/", "XDG_RUNTIME_DIR": "/run/user/1234"},
            {"HOME": "/srv/test-user", "XDG_RUNTIME_DIR": "/run/../tmp"},
            {"HOME": "/srv/test-user", "XDG_RUNTIME_DIR": "/run/user/1234\nother"},
        )
        for environment in invalid_environments:
            with self.subTest(environment=environment), self.assertRaises(RuntimeContractError):
                RuntimePaths.from_environment(environment, uid=1234)
        with self.assertRaises(RuntimeContractError):
            RuntimePaths.from_environment({"HOME": "/srv/test-user"}, uid=True)

    def test_private_directory_requires_owner_and_exact_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private"
            path.mkdir(mode=0o700)
            self.assertEqual(
                private_directory_status(path, uid=os.getuid()),
                (True, "owner-only directory"),
            )
            path.chmod(0o750)
            ok, detail = private_directory_status(path, uid=os.getuid())
            self.assertFalse(ok)
            self.assertIn("0700", detail)


if __name__ == "__main__":
    unittest.main()
