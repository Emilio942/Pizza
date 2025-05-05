from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from .database import Database, TaskManager
from .models import TaskStatus

app = FastAPI(title="ChatList-KI API")
db = Database()
task_manager = TaskManager(db)

class TaskCreate(BaseModel):
    title: str
    priority: int
    task_id: Optional[str] = None

class TaskUpdate(BaseModel):
    status: Optional[TaskStatus] = None
    priority: Optional[int] = None
    due_date: Optional[datetime] = None
    notes: Optional[str] = None

@app.post("/tasks/")
async def create_task(task: TaskCreate):
    created_task = task_manager.add_task(
        title=task.title,
        priority=task.priority,
        task_id=task.task_id
    )
    return created_task.to_dict()

@app.get("/tasks/")
async def list_tasks(priority: Optional[int] = None, status: Optional[TaskStatus] = None):
    tasks = task_manager.get_tasks(priority=priority, status=status)
    return [task.to_dict() for task in tasks]

@app.patch("/tasks/{task_id}")
async def update_task(task_id: str, task_update: TaskUpdate):
    task = None
    if task_update.status is not None:
        task = task_manager.update_task_status(task_id, task_update.status)
    if task_update.priority is not None:
        task = task_manager.update_task_priority(task_id, task_update.priority)
    if task_update.due_date is not None:
        task = task_manager.set_due_date(task_id, task_update.due_date)
    if task_update.notes is not None:
        task = task_manager.add_notes(task_id, task_update.notes)
    
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.to_dict()

@app.get("/tasks/priority/{priority}")
async def get_tasks_by_priority(priority: int):
    tasks = task_manager.get_tasks(priority=priority)
    return [task.to_dict() for task in tasks]