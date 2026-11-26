import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import database  # noqa: E402
import seed_data  # noqa: E402
from rag_engine import HashingEmbedder, RAGEngine  # noqa: E402


class FakeGenerator:
    """Stand-in for Ollama so RAG tests are deterministic and offline."""
    def __init__(self, reply="", is_available=True):
        self.reply, self.is_available, self.calls = reply, is_available, []

    def available(self):
        return self.is_available

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.reply


@pytest.fixture()
def db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield tmp_path
    database.configure(tmp_path / "unused.db")


@pytest.fixture()
def seeded(db):
    seed_data.seed_if_empty()
    return db


@pytest.fixture()
def engine(seeded):
    return RAGEngine(embedder=HashingEmbedder(), vector_dir=seeded / "vs", generator=FakeGenerator())


@pytest.fixture()
def user(db):
    return database.create_user("tester", "tester@example.com", "x")


VALID = dict(title="ModuleNotFoundError: No module named requests", technology="Python", version="3.12",
             operating_system="Windows 11", category="Environment / Package Error",
             error_message="ModuleNotFoundError: No module named 'requests'",
             description="requests is installed but my script cannot import it.",
             root_cause="The script runs in a different virtual environment.",
             working_solution="Activate the right venv and run python -m pip install requests.")
