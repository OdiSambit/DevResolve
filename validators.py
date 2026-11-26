"""Input validation, sanitisation and secret detection."""
import re

import config


class ValidationError(Exception):
    def __init__(self, errors):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MIN_LEN = {"title": 8, "error_message": 5, "description": 20, "root_cause": 10, "working_solution": 10}
MAX_LEN = {"title": 200, "version": 40, "error_message": 4000, "description": 4000,
           "root_cause": 3000, "working_solution": 4000}
LABELS = {"title": "Problem title", "technology": "Technology", "version": "Technology version",
          "operating_system": "Operating system", "category": "Error category",
          "error_message": "Exact error message", "description": "Problem description",
          "root_cause": "Root cause", "working_solution": "Working solution"}


def sanitize_text(text) -> str:
    """Strip control characters / normalise newlines. Display code must still escape output."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub("", text).strip()


# High-confidence credential patterns => block the submission
_BLOCK_PATTERNS = [
    (r"sk-[A-Za-z0-9_\-]{20,}", "an API key (sk-...)"),
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "a GitHub token"),
    (r"AKIA[0-9A-Z]{16}", "an AWS access key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "a private key"),
    (r"xox[baprs]-[A-Za-z0-9\-]{10,}", "a Slack token"),
]
# Lower-confidence hints => warn only
_WARN_PATTERNS = [
    (r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\s*[=:]\s*\S{4,}", "a password/secret assignment"),
    (r"(?i)bearer\s+[A-Za-z0-9\-_.=]{20,}", "a bearer token"),
]


def detect_secrets(text: str):
    """Returns (blocking, warnings) lists of human-readable descriptions."""
    blocking = [d for p, d in _BLOCK_PATTERNS if re.search(p, text or "")]
    warnings = [d for p, d in _WARN_PATTERNS if re.search(p, text or "")]
    return blocking, warnings


def _meaningful_error(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    return letters >= 3 and len(set(text.lower().replace(" ", ""))) >= 3


def validate_experience(data: dict) -> dict:
    errors, cleaned = [], {}
    for field in LABELS:
        cleaned[field] = sanitize_text(data.get(field, ""))

    for field in ("title", "technology", "operating_system", "category",
                  "error_message", "description", "root_cause", "working_solution"):
        if not cleaned[field]:
            errors.append(f"{LABELS[field]} is required.")

    for field, n in MIN_LEN.items():
        if cleaned[field] and len(cleaned[field]) < n:
            errors.append(f"{LABELS[field]} is too short (minimum {n} characters).")
    for field, n in MAX_LEN.items():
        if len(cleaned[field]) > n:
            errors.append(f"{LABELS[field]} is too long (maximum {n} characters).")

    if cleaned["error_message"] and len(cleaned["error_message"]) >= 5 and not _meaningful_error(cleaned["error_message"]):
        errors.append("Exact error message does not look meaningful - paste the real error text.")

    if cleaned["technology"] and cleaned["technology"] not in config.TECHNOLOGIES:
        errors.append("Technology must be one of the supported categories.")
    if cleaned["operating_system"] and cleaned["operating_system"] not in config.OPERATING_SYSTEMS:
        errors.append("Operating system must be one of the supported options.")
    if cleaned["category"] and cleaned["category"] not in config.CATEGORIES:
        errors.append("Error category must be one of the supported options.")

    combined = "\n".join(cleaned.values())
    blocking, _ = detect_secrets(combined)
    for item in blocking:
        errors.append(f"Your submission appears to contain {item}. Remove or redact it before posting.")

    if errors:
        raise ValidationError(errors)
    return cleaned


def validate_solution(text: str) -> str:
    text = sanitize_text(text)
    if len(text) < 10:
        raise ValidationError("Solution is too short (minimum 10 characters).")
    if len(text) > 4000:
        raise ValidationError("Solution is too long (maximum 4000 characters).")
    blocking, _ = detect_secrets(text)
    if blocking:
        raise ValidationError(f"Your solution appears to contain {blocking[0]}. Remove or redact it.")
    return text
