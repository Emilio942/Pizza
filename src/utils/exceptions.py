"""
Benutzerdefinierte Ausnahmen für das Pizza-Erkennungssystem.
"""

class PizzaDetectorError(Exception):
    """Basisklasse für alle Pizza-Detektor Ausnahmen."""
    pass

class InvalidInputError(PizzaDetectorError):
    """Ausnahme für ungültige Eingaben."""
    pass

class ConfigError(PizzaDetectorError):
    """Ausnahme für Konfigurationsfehler."""
    pass

class ModelError(PizzaDetectorError):
    """Ausnahme für Modellfehler."""
    pass

class HardwareError(PizzaDetectorError):
    """Ausnahme für Hardware-Fehler."""
    pass

class ResourceError(PizzaDetectorError):
    """Ausnahme für Ressourcenüberschreitungen."""
    pass

class DataError(PizzaDetectorError):
    """Ausnahme für Datenfehler."""
    pass

class QuantizationError(PizzaDetectorError):
    """Ausnahme für Quantisierungsfehler."""
    pass

class ExportError(PizzaDetectorError):
    """Ausnahme für Exportfehler."""
    pass

class EmulatorError(PizzaDetectorError):
    """Ausnahme für Emulator-Fehler."""
    pass

# Ausnahmen für spezifische Hardware-Fehler
class CameraError(HardwareError):
    """Ausnahme für Kamerafehler."""
    pass

class MemoryError(HardwareError):
    """Ausnahme für Speicherfehler."""
    pass

class PowerError(HardwareError):
    """Ausnahme für Energiefehler."""
    pass

class CommunicationError(HardwareError):
    """Ausnahme für Kommunikationsfehler."""
    pass

# Ausnahmen für spezifische Modell-Fehler
class ModelSizeError(ModelError):
    """Ausnahme für Modellgrößenfehler."""
    pass

class ModelCompatibilityError(ModelError):
    """Ausnahme für Modellkompatibilitätsfehler."""
    pass

class InferenceError(ModelError):
    """Ausnahme für Inferenzfehler."""
    pass

# Ausnahmen für spezifische Ressourcenfehler
class RAMOverflowError(ResourceError):
    """Ausnahme für RAM-Überlauf."""
    pass

class FlashOverflowError(ResourceError):
    """Ausnahme für Flash-Überlauf."""
    pass

# Ausnahmen für spezifische Datenfehler
class DatasetError(DataError):
    """Ausnahme für Datensatzfehler."""
    pass

class FileOperationError(DataError):
    """Ausnahme für Dateioperationsfehler."""
    pass

# Hilfreiche Funktionen für Ausnahmebehandlung
def wrap_hardware_errors(func):
    """Decorator für Hardware-Fehlerbehandlung."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HardwareError:
            raise  # Originalen HardwareError weiterleiten
        except Exception as e:
            raise HardwareError(f"Hardware-Fehler in {func.__name__}: {str(e)}") from e
    return wrapper

def wrap_model_errors(func):
    """Decorator für Modell-Fehlerbehandlung."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelError:
            raise  # Originalen ModelError weiterleiten
        except Exception as e:
            raise ModelError(f"Modellfehler in {func.__name__}: {str(e)}") from e
    return wrapper

def wrap_resource_errors(func):
    """Decorator für Ressourcen-Fehlerbehandlung."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ResourceError:
            raise  # Originalen ResourceError weiterleiten
        except Exception as e:
            raise ResourceError(f"Ressourcenfehler in {func.__name__}: {str(e)}") from e
    return wrapper