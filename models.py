"""SQLAlchemy models: users, experiences, solutions and knowledge records."""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Experience(Base):
    __tablename__ = "experiences"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    technology: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(40), default="")
    operating_system: Mapped[str] = mapped_column(String(50), nullable=False)
    category: Mapped[str] = mapped_column(String(60), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    working_solution: Mapped[str] = mapped_column(Text, nullable=False)   # author-reported fix
    is_seed: Mapped[bool] = mapped_column(Boolean, default=False)         # sample/seeded record
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Solution(Base):
    __tablename__ = "solutions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experience_id: Mapped[int] = mapped_column(ForeignKey("experiences.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    solution: Mapped[str] = mapped_column(Text, nullable=False)
    is_accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeRecord(Base):
    """Maps indexed content to its source experience. faiss_id NULL => pending (re-)indexing."""
    __tablename__ = "knowledge_records"
    __table_args__ = (UniqueConstraint("source_id", "source_type"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_type: Mapped[str] = mapped_column(String(30), default="experience")
    technology: Mapped[str] = mapped_column(String(50), default="")
    version: Mapped[str] = mapped_column(String(40), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    faiss_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
