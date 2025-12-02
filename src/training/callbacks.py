import logging
import torch

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Verbesserte Early-Stopping-Implementation mit Validation Loss Plateau-Erkennung"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.val_loss_history = []
    
    def __call__(self, val_loss, model):
        # Speichere alle Validierungsverluste
        self.val_loss_history.append(val_loss)
        
        score = -val_loss
        
        if self.best_score is None:
            # Erster Aufruf
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            # Verschlechterung oder Stagnation
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            
            # Prüfe auf Plateau oder Divergenz
            if len(self.val_loss_history) >= 5:
                # Berechne gleitenden Durchschnitt der letzten 3 Verluste
                recent_avg = sum(self.val_loss_history[-3:]) / 3
                # Wenn der Verlust konstant oder steigend ist
                if all(l >= self.val_loss_history[-4] for l in self.val_loss_history[-3:]):
                    logger.info("Plateau oder steigender Validierungsverlust erkannt")
                    self.counter = max(self.counter, self.patience - 2)  # Beschleunige Abbruch
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Verbesserung
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def restore_weights(self, model):
        """Stellt die besten Gewichte wieder her"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            return True
        return False
