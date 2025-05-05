from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from .models import Base, Task, TaskStatus
from typing import List, Optional
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

class Database:
    def __init__(self, db_url: str = None):
        load_dotenv()
        self.engine = create_engine(
            db_url or os.getenv("DATABASE_URL", "sqlite:///chatlist_ki.db")
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

class TaskManager:
    def __init__(self, db: Database):
        self.db = db

    def add_task(self, title: str, priority: int, task_id: Optional[str] = None) -> Task:
        with self.db.get_session() as session:
            if task_id is None:
                # Get the highest task number and increment
                last_task = session.query(Task).order_by(desc(Task.task_id)).first()
                if last_task:
                    last_num = int(last_task.task_id[1:])
                    task_id = f"T{last_num + 1}"
                else:
                    task_id = "T1"

            task = Task(
                task_id=task_id,
                title=title,
                priority=priority,
                status=TaskStatus.OPEN
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

    def get_tasks(self, priority: Optional[int] = None, status: Optional[TaskStatus] = None) -> List[Task]:
        with self.db.get_session() as session:
            query = session.query(Task)
            if priority is not None:
                query = query.filter(Task.priority == priority)
            if status is not None:
                query = query.filter(Task.status == status)
            return query.order_by(Task.priority, Task.created_at).all()

    def update_task_status(self, task_id: str, status: TaskStatus) -> Optional[Task]:
        with self.db.get_session() as session:
            task = session.query(Task).filter(Task.task_id == task_id.upper()).first()
            if task:
                task.status = status
                task.updated_at = datetime.now(timezone.UTC)
                session.commit()
                session.refresh(task)
                return task
            return None

    def update_task_priority(self, task_id: str, priority: int) -> Optional[Task]:
        with self.db.get_session() as session:
            task = session.query(Task).filter(Task.task_id == task_id.upper()).first()
            if task:
                task.priority = priority
                task.updated_at = datetime.now(timezone.UTC)
                session.commit()
                session.refresh(task)
                return task
            return None

    def set_due_date(self, task_id: str, due_date: datetime) -> Optional[Task]:
        with self.db.get_session() as session:
            task = session.query(Task).filter(Task.task_id == task_id.upper()).first()
            if task:
                task.due_date = due_date
                task.updated_at = datetime.now(timezone.UTC)
                session.commit()
                session.refresh(task)
                return task
            return None

    def add_notes(self, task_id: str, notes: str) -> Optional[Task]:
        with self.db.get_session() as session:
            task = session.query(Task).filter(Task.task_id == task_id.upper()).first()
            if task:
                task.notes = notes
                task.updated_at = datetime.now(timezone.UTC)
                session.commit()
                session.refresh(task)
                return task
            return None