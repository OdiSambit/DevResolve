"""SQLite persistence layer. All public functions return plain dicts (no detached ORM objects)."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, event, func, or_, select
from sqlalchemy.orm import sessionmaker

import config
from models import Base, Experience, KnowledgeRecord, Solution, User
from validators import ValidationError, validate_experience, validate_solution

_engine = None
_Session = None


def configure(db_path=None):
    """(Re)connect to a SQLite file and create tables. Call with a temp path in tests."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(_engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    return _engine


init_db = configure


@contextmanager
def session_scope():
    if _Session is None:
        configure()
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ------------------------------------------------------------------ users
def _user_dict(u):
    return {"id": u.id, "username": u.username, "email": u.email,
            "password_hash": u.password_hash, "created_at": u.created_at}


def create_user(username, email, password_hash):
    with session_scope() as s:
        u = User(username=username, email=email.lower(), password_hash=password_hash)
        s.add(u)
        s.flush()
        return _user_dict(u)


def get_user_by_username(username):
    with session_scope() as s:
        u = s.scalar(select(User).where(func.lower(User.username) == username.lower()))
        return _user_dict(u) if u else None


def get_user_by_email(email):
    with session_scope() as s:
        u = s.scalar(select(User).where(User.email == email.lower()))
        return _user_dict(u) if u else None


# ------------------------------------------------------------- experiences
def _summaries(s, rows):
    """Convert Experience rows to dicts with author, solution count and accepted flag."""
    ids = [e.id for e in rows]
    authors = {u.id: u.username for u in s.scalars(select(User).where(User.id.in_({e.user_id for e in rows})))} if rows else {}
    counts, accepted = {}, set()
    if ids:
        for eid, n in s.execute(select(Solution.experience_id, func.count()).where(Solution.experience_id.in_(ids)).group_by(Solution.experience_id)):
            counts[eid] = n
        for (eid,) in s.execute(select(Solution.experience_id).where(Solution.experience_id.in_(ids), Solution.is_accepted.is_(True))):
            accepted.add(eid)
    out = []
    for e in rows:
        out.append({
            "id": e.id, "user_id": e.user_id, "author": authors.get(e.user_id, "unknown"),
            "title": e.title, "technology": e.technology, "version": e.version or "",
            "operating_system": e.operating_system, "category": e.category,
            "error_message": e.error_message, "description": e.description,
            "root_cause": e.root_cause, "working_solution": e.working_solution,
            "is_seed": bool(e.is_seed), "created_at": e.created_at,
            "solution_count": counts.get(e.id, 0), "has_accepted": e.id in accepted,
        })
    return out


def create_experience(user_id, data, is_seed=False, validate=True):
    cleaned = validate_experience(data) if validate else data
    with session_scope() as s:
        e = Experience(user_id=user_id, is_seed=is_seed, **{k: cleaned.get(k, "") for k in (
            "title", "technology", "version", "operating_system", "category",
            "error_message", "description", "root_cause", "working_solution")})
        s.add(e)
        s.flush()
        return _summaries(s, [e])[0]


def list_experiences(technology=None, category=None, operating_system=None, query=None,
                     user_id=None, limit=50):
    with session_scope() as s:
        stmt = select(Experience).order_by(Experience.created_at.desc(), Experience.id.desc())
        if technology:
            stmt = stmt.where(Experience.technology == technology)
        if category:
            stmt = stmt.where(Experience.category == category)
        if operating_system:
            stmt = stmt.where(Experience.operating_system == operating_system)
        if user_id:
            stmt = stmt.where(Experience.user_id == user_id)
        if query and query.strip():
            like = f"%{query.strip()}%"
            stmt = stmt.where(or_(Experience.title.ilike(like), Experience.error_message.ilike(like),
                                  Experience.description.ilike(like), Experience.root_cause.ilike(like)))
        rows = s.scalars(stmt.limit(limit)).all()
        return _summaries(s, rows)


def get_experience(experience_id):
    with session_scope() as s:
        e = s.get(Experience, experience_id)
        if not e:
            return None
        d = _summaries(s, [e])[0]
        sols = s.execute(select(Solution, User.username).join(User, User.id == Solution.user_id)
                         .where(Solution.experience_id == experience_id)
                         .order_by(Solution.is_accepted.desc(), Solution.created_at.asc())).all()
        d["solutions"] = [{"id": sol.id, "solution": sol.solution, "author": name,
                           "user_id": sol.user_id, "is_accepted": bool(sol.is_accepted),
                           "created_at": sol.created_at} for sol, name in sols]
        return d


def get_experiences_by_ids(ids):
    """Full records (with solutions) in the same order as `ids`; unknown ids are skipped."""
    out = []
    for i in ids:
        d = get_experience(int(i))
        if d:
            out.append(d)
    return out


def all_experience_ids():
    with session_scope() as s:
        return [i for (i,) in s.execute(select(Experience.id).order_by(Experience.id))]


def count_experiences():
    with session_scope() as s:
        return s.scalar(select(func.count()).select_from(Experience)) or 0


# ---------------------------------------------------------------- solutions
def add_solution(experience_id, user_id, text, is_accepted=False):
    text = validate_solution(text)
    with session_scope() as s:
        if not s.get(Experience, experience_id):
            raise ValidationError("Experience not found.")
        sol = Solution(experience_id=experience_id, user_id=user_id, solution=text, is_accepted=is_accepted)
        s.add(sol)
        s.flush()
        return {"id": sol.id, "experience_id": experience_id}


def accept_solution(solution_id, acting_user_id):
    """Only the experience author may accept; exactly one solution is accepted at a time."""
    with session_scope() as s:
        sol = s.get(Solution, solution_id)
        if not sol:
            raise ValidationError("Solution not found.")
        exp = s.get(Experience, sol.experience_id)
        if exp.user_id != acting_user_id:
            raise ValidationError("Only the author of the experience can accept a solution.")
        for other in s.scalars(select(Solution).where(Solution.experience_id == exp.id)):
            other.is_accepted = other.id == solution_id
        return exp.id


# --------------------------------------------------------- knowledge records
def upsert_knowledge_record(exp_id, technology, version, content):
    """Create/update the knowledge document; resets faiss_id so it counts as pending."""
    with session_scope() as s:
        rec = s.scalar(select(KnowledgeRecord).where(KnowledgeRecord.source_id == exp_id,
                                                     KnowledgeRecord.source_type == "experience"))
        if rec is None:
            rec = KnowledgeRecord(source_id=exp_id, source_type="experience")
            s.add(rec)
        rec.technology, rec.version, rec.content, rec.faiss_id = technology, version or "", content, None


def mark_indexed(exp_id, faiss_id):
    with session_scope() as s:
        rec = s.scalar(select(KnowledgeRecord).where(KnowledgeRecord.source_id == exp_id,
                                                     KnowledgeRecord.source_type == "experience"))
        if rec:
            rec.faiss_id = faiss_id


def experiences_needing_index():
    """Experience ids with no knowledge record yet, or whose record is pending (faiss_id NULL)."""
    with session_scope() as s:
        indexed = {sid for (sid,) in s.execute(select(KnowledgeRecord.source_id).where(
            KnowledgeRecord.faiss_id.is_not(None), KnowledgeRecord.source_type == "experience"))}
        return [i for i in all_experience_ids() if i not in indexed]


def indexed_experience_ids():
    with session_scope() as s:
        return sorted(sid for (sid,) in s.execute(select(KnowledgeRecord.source_id).where(
            KnowledgeRecord.faiss_id.is_not(None), KnowledgeRecord.source_type == "experience")))


def get_knowledge_content(exp_id):
    with session_scope() as s:
        rec = s.scalar(select(KnowledgeRecord).where(KnowledgeRecord.source_id == exp_id))
        return rec.content if rec else None
