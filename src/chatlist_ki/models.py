from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TaskStatus(str, Enum):
    OPEN = "Offen"
    IN_PROGRESS = "In Arbeit"
    COMPLETED = "Abgeschlossen"
    BLOCKED = "Blockiert"

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(10), unique=True, nullable=False)  # e.g., T1, T2, etc.
    title = Column(String(255), nullable=False)
    priority = Column(Integer, nullable=False)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.OPEN)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    due_date = Column(DateTime, nullable=True)
    notes = Column(String(1000), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "title": self.title,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "notes": self.notes
        }