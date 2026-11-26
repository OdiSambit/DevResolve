import pytest

import auth
import database
from conftest import VALID
from validators import ValidationError, detect_secrets, sanitize_text


def test_submit_valid_experience(user):
    exp = database.create_experience(user["id"], VALID)
    assert exp["id"] and database.get_experience(exp["id"])["title"] == VALID["title"]


@pytest.mark.parametrize("field", ["title", "error_message", "description", "root_cause", "working_solution"])
def test_required_fields_rejected_when_empty(user, field):
    with pytest.raises(ValidationError):
        database.create_experience(user["id"], {**VALID, field: ""})


def test_too_short_and_meaningless_rejected(user):
    with pytest.raises(ValidationError):
        database.create_experience(user["id"], {**VALID, "description": "short"})
    with pytest.raises(ValidationError):
        database.create_experience(user["id"], {**VALID, "error_message": "!!!!!!!!"})


def test_invalid_dropdown_values_rejected(user):
    with pytest.raises(ValidationError):
        database.create_experience(user["id"], {**VALID, "technology": "Cobol"})


def test_secrets_are_blocked_and_hints_warn(user):
    bad = {**VALID, "error_message": "auth failed with key sk-abcdefghijklmnopqrstuvwxyz123456"}
    with pytest.raises(ValidationError):
        database.create_experience(user["id"], bad)
    blocking, warnings = detect_secrets("password=hunter2hunter2")
    assert not blocking and warnings


def test_sanitize_strips_control_chars():
    assert sanitize_text("a\x00b\x07c\r\nd ") == "abc\nd"


def test_feed_lists_seed_data(seeded):
    exps = database.list_experiences(limit=100)
    assert len(exps) == 47 >= 30
    assert all(e["is_seed"] for e in exps)
    assert {e["technology"] for e in database.list_experiences(technology="Docker")} == {"Docker"}
    assert any("pandas" in e["title"] for e in database.list_experiences(query="pandas"))


def test_solutions_and_accepting(seeded, user):
    other = database.create_user("other", "o@example.com", "x")
    exp = database.create_experience(user["id"], VALID)
    sol = database.add_solution(exp["id"], other["id"], "Try python -m pip install requests in the venv.")
    with pytest.raises(ValidationError):                       # only the author may accept
        database.accept_solution(sol["id"], other["id"])
    database.accept_solution(sol["id"], user["id"])
    detail = database.get_experience(exp["id"])
    assert detail["has_accepted"] and detail["solutions"][0]["is_accepted"]
    assert detail["solution_count"] == 1


def test_only_one_accepted_solution(user):
    exp = database.create_experience(user["id"], VALID)
    a = database.add_solution(exp["id"], user["id"], "First possible solution text.")["id"]
    b = database.add_solution(exp["id"], user["id"], "Second possible solution text.")["id"]
    database.accept_solution(a, user["id"])
    database.accept_solution(b, user["id"])
    accepted = [s for s in database.get_experience(exp["id"])["solutions"] if s["is_accepted"]]
    assert [s["id"] for s in accepted] == [b]


def test_records_persist_after_restart(db, user):
    exp = database.create_experience(user["id"], VALID)
    database.configure(db / "test.db")                          # simulate app restart
    assert database.get_experience(exp["id"])["title"] == VALID["title"]


def test_auth_hashes_passwords_and_logs_in(db):
    u = auth.register("alice_dev", "alice@example.com", "correct-horse-1")
    stored = database.get_user_by_username("alice_dev")["password_hash"]
    assert stored != "correct-horse-1" and stored.startswith("$2")
    assert auth.login("alice@example.com", "correct-horse-1")["id"] == u["id"]
    with pytest.raises(ValidationError):
        auth.login("alice_dev", "wrong-password")
    with pytest.raises(ValidationError):
        auth.register("alice_dev", "other@example.com", "correct-horse-1")   # duplicate username
    with pytest.raises(ValidationError):
        auth.register("bob", "bad-email", "short")
