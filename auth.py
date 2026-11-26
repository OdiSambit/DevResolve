"""Minimal registration / login using bcrypt password hashing."""
import re

import bcrypt

import database
from validators import ValidationError

_USERNAME = re.compile(r"^[A-Za-z0-9_\-]{3,30}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except ValueError:
        return False


def register(username: str, email: str, password: str) -> dict:
    username, email = (username or "").strip(), (email or "").strip().lower()
    errors = []
    if not _USERNAME.match(username):
        errors.append("Username must be 3-30 characters: letters, numbers, _ or -.")
    if not _EMAIL.match(email):
        errors.append("Enter a valid email address.")
    if len(password or "") < 8:
        errors.append("Password must be at least 8 characters.")
    if errors:
        raise ValidationError(errors)
    if database.get_user_by_username(username):
        raise ValidationError("That username is already taken.")
    if database.get_user_by_email(email):
        raise ValidationError("That email is already registered.")
    user = database.create_user(username, email, hash_password(password))
    return _public(user)


def login(identifier: str, password: str) -> dict:
    identifier = (identifier or "").strip()
    user = database.get_user_by_username(identifier) or database.get_user_by_email(identifier)
    if not user or not verify_password(password or "", user["password_hash"]):
        raise ValidationError("Invalid username/email or password.")
    return _public(user)


def _public(user: dict) -> dict:
    return {"id": user["id"], "username": user["username"], "email": user["email"]}
