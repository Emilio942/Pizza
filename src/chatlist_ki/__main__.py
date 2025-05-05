from .chat_interface import ChatInterface
from .database import Database, TaskManager
from .models import TaskStatus

def initialize_tasks():
    """Initialize the system with predefined tasks from the design document."""
    db = Database()
    task_manager = TaskManager(db)
    
    initial_tasks = [
        ("NSFW-Erkennungsfilter optimieren", 1),
        ("RAM-Nutzung reduzieren", 1),
        ("C-Code-Export-Pipeline aufsetzen", 1),
        ("Batterie-Benchmark & Laufzeitmessung", 2),
        ("Datenkategorien (verbrannte Klassen) erweitern", 2),
        ("Feldtests in verschiedenen Umgebungen", 1),
        ("Modularen API-Endpunkt designen", 1),
        ("Pruning-Strategie prototypisch umsetzen", 3),
    ]
    
    for title, priority in initial_tasks:
        task_manager.add_task(title=title, priority=priority)

def main():
    """Main CLI interface for ChatList-KI."""
    print("Willkommen bei ChatList-KI!")
    print("Tippe 'hilfe' für verfügbare Befehle oder 'beenden' zum Beenden.")
    print("Initialisiere Aufgabenliste...")
    
    try:
        initialize_tasks()
        print("Aufgabenliste initialisiert!")
    except Exception as e:
        print(f"Fehler beim Initialisieren der Aufgaben: {e}")
    
    chat_interface = ChatInterface()
    
    while True:
        try:
            user_input = input("\nChatList-KI> ").strip()
            
            if user_input.lower() == 'beenden':
                print("Auf Wiedersehen!")
                break
                
            if user_input.lower() == 'hilfe':
                print("""
Verfügbare Befehle:
- zeige alle tasks
- zeige tasks mit priorität X
- neue aufgabe: [Titel] mit priorität X
- markiere TX als (in arbeit|abgeschlossen|blockiert)
- ändere priorität von TX auf Y
- setze fälligkeitsdatum für TX auf DD.MM.YYYY
- beenden
                """)
                continue
                
            if user_input:
                response = chat_interface.process_command(user_input)
                print("\n" + response)
                
        except KeyboardInterrupt:
            print("\nAuf Wiedersehen!")
            break
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()