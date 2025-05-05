import pytest
import os
import sys
from datetime import datetime
from src.chatlist_ki import ChatInterface, TaskStatus, Database, TaskManager

@pytest.fixture
def chat_interface():
    # Use an in-memory SQLite database for testing
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    return ChatInterface()

def test_add_task(chat_interface):
    response = chat_interface.process_command("neue aufgabe: Test Task mit priorität 1")
    assert "test task mit priorität 1" in response.lower()
    assert "priorität: 1" in response.lower()

def test_show_tasks_by_priority(chat_interface):
    # First add some tasks
    chat_interface.process_command("neue aufgabe: P1 Task mit priorität 1")
    chat_interface.process_command("neue aufgabe: P2 Task mit priorität 2")
    chat_interface.process_command("neue aufgabe: Another P1 mit priorität 1")
    
    # Test filtering by priority
    response = chat_interface.process_command("zeige tasks mit priorität 1")
    assert "p1 task" in response.lower()
    assert "another p1" in response.lower()
    assert "p2 task" not in response.lower()

def test_update_task_status(chat_interface):
    # Add a task first
    chat_interface.process_command("neue aufgabe: Status Test mit priorität 1")
    
    # Update its status
    response = chat_interface.process_command("markiere T1 als in arbeit")
    assert "T1 ist jetzt auf **In Arbeit** gesetzt" in response
    
    # Verify in task list
    task_list = chat_interface.process_command("zeige alle tasks")
    assert "status test" in task_list.lower()
    assert "◔" in task_list  # Status marker for "in arbeit"

def test_update_priority(chat_interface):
    chat_interface.process_command("neue aufgabe: Priority Test mit priorität 1")
    response = chat_interface.process_command("ändere priorität von T1 auf 3")
    assert "Priorität von T1 auf 3 geändert" in response

def test_set_due_date(chat_interface):
    chat_interface.process_command("neue aufgabe: Due Date Test mit priorität 1")
    response = chat_interface.process_command("setze fälligkeitsdatum für T1 auf 01.06.2025")
    assert "Fälligkeitsdatum für T1 auf 01.06.2025 gesetzt" in response
    
    # Verify in task list
    task_list = chat_interface.process_command("zeige alle tasks")
    assert "01.06.2025" in task_list

def test_invalid_commands(chat_interface):
    response = chat_interface.process_command("ungültiger befehl")
    assert "Entschuldigung, ich habe den Befehl nicht verstanden" in response
    assert "Verfügbare Befehle:" in response

def test_task_status_workflow(chat_interface):
    # Add task
    chat_interface.process_command("neue aufgabe: Workflow Test mit priorität 1")
    
    # Test different status transitions
    statuses = [
        ("in arbeit", "◔"),
        ("blockiert", "⊘"),
        ("abgeschlossen", "●"),
    ]
    
    for status, marker in statuses:
        chat_interface.process_command(f"markiere T1 als {status}")
        task_list = chat_interface.process_command("zeige alle tasks")
        assert marker in task_list

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    print("Current directory:", os.getcwd())
    
    try:
        print("\nTesting imports...")
        from src.chatlist_ki import ChatInterface
        print("ChatInterface import successful")
        
        print("\nTesting initialization...")
        chat = ChatInterface()
        print("ChatInterface initialization successful")
        
        print("\nTesting command processing...")
        response = chat.process_command("zeige alle tasks")
        print("Command response:", response)
        
        print("\nAll manual tests passed!")
        
    except Exception as e:
        print("\nError during testing:")
        print(f"Type: {type(e)}")
        print(f"Message: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()