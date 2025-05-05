from typing import List, Optional, Dict, Any
from .database import TaskManager, Database
from .models import TaskStatus
import re
from datetime import datetime, timedelta

class ChatInterface:
    def __init__(self):
        self.db = Database()
        self.task_manager = TaskManager(self.db)
        
    def process_command(self, user_input: str) -> str:
        """Process natural language commands and return appropriate responses."""
        user_input = user_input.lower().strip()
        
        # Show tasks with specific priority
        if match := re.search(r"zeige (?:mir )?(?:alle )?tasks? mit priorität (\d+)", user_input):
            priority = int(match.group(1))
            tasks = self.task_manager.get_tasks(priority=priority)
            if not tasks:
                return f"Keine Tasks mit Priorität {priority} gefunden."
            return self._format_task_list(tasks)
            
        # Mark task as in progress/completed
        if match := re.search(r"markiere (?:task )?([tT]\d+) als (in arbeit|abgeschlossen|blockiert)", user_input):
            task_id = match.group(1).upper()
            status_text = match.group(2)
            status_map = {
                "in arbeit": TaskStatus.IN_PROGRESS,
                "abgeschlossen": TaskStatus.COMPLETED,
                "blockiert": TaskStatus.BLOCKED
            }
            status = status_map[status_text]
            task = self.task_manager.update_task_status(task_id, status)
            if task:
                return f"Task {task_id} ist jetzt auf **{status.value}** gesetzt."
            return f"Task {task_id} nicht gefunden."

        # Create new task
        if "neue aufgabe" in user_input or "neuer task" in user_input:
            # Extract priority if specified
            priority = 3  # Default priority
            if match := re.search(r"priorität[: ](\d+)", user_input):
                priority = int(match.group(1))
            
            # Extract title - everything after ":" or the last occurrence of "mit priorität X"
            title = user_input
            if ":" in user_input:
                title = user_input.split(":", 1)[1].strip()
            elif "mit priorität" in user_input:
                title = re.sub(r"mit priorität \d+", "", user_input)
                title = re.sub(r"neue[rn]? (?:aufgabe|task)", "", title).strip()
            else:
                title = re.sub(r"neue[rn]? (?:aufgabe|task)", "", title).strip()
                
            task = self.task_manager.add_task(title=title, priority=priority)
            return f"Neue Aufgabe erstellt: {task.task_id} - {task.title} (Priorität: {task.priority})"

        # Show all tasks
        if "zeige alle tasks" in user_input or "liste alle aufgaben" in user_input:
            tasks = self.task_manager.get_tasks()
            if not tasks:
                return "Keine Tasks vorhanden."
            return self._format_task_list(tasks)

        # Set due date
        if match := re.search(r"setze fälligkeitsdatum für ([tT]\d+) auf (.+)", user_input):
            task_id = match.group(1).upper()
            date_text = match.group(2)
            try:
                # Simple date parsing - extend as needed
                if "heute" in date_text:
                    due_date = datetime.now()
                elif "morgen" in date_text:
                    due_date = datetime.now() + timedelta(days=1)
                else:
                    # Expect format DD.MM.YYYY
                    due_date = datetime.strptime(date_text, "%d.%m.%Y")
                
                task = self.task_manager.set_due_date(task_id, due_date)
                if task:
                    return f"Fälligkeitsdatum für {task_id} auf {due_date.strftime('%d.%m.%Y')} gesetzt."
                return f"Task {task_id} nicht gefunden."
            except ValueError:
                return "Ungültiges Datumsformat. Bitte verwende DD.MM.YYYY oder 'heute'/'morgen'."

        # Update priority
        if match := re.search(r"ändere priorität von ([tT]\d+) auf (\d+)", user_input):
            task_id = match.group(1).upper()
            priority = int(match.group(2))
            task = self.task_manager.update_task_priority(task_id, priority)
            if task:
                return f"Priorität von {task_id} auf {priority} geändert."
            return f"Task {task_id} nicht gefunden."

        return "Entschuldigung, ich habe den Befehl nicht verstanden. Verfügbare Befehle:\n" + \
               "- Zeige alle Tasks\n" + \
               "- Zeige Tasks mit Priorität X\n" + \
               "- Neue Aufgabe: [Titel] mit Priorität X\n" + \
               "- Markiere TX als (in arbeit|abgeschlossen|blockiert)\n" + \
               "- Ändere Priorität von TX auf Y\n" + \
               "- Setze Fälligkeitsdatum für TX auf DD.MM.YYYY"

    def _format_task_list(self, tasks: List[Any]) -> str:
        """Format a list of tasks for display."""
        if not tasks:
            return "Keine Tasks gefunden."
            
        result = []
        for task in tasks:
            status_marker = {
                TaskStatus.OPEN: "○",
                TaskStatus.IN_PROGRESS: "◔",
                TaskStatus.COMPLETED: "●",
                TaskStatus.BLOCKED: "⊘"
            }.get(task.status, "○")
            
            due_date = f" (Fällig: {task.due_date.strftime('%d.%m.%Y')})" if task.due_date else ""
            result.append(f"{status_marker} {task.task_id}: {task.title} (P{task.priority}){due_date}")
            
        return "\n".join(result)