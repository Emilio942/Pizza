#!/usr/bin/env python3
# hardware/manufacturing/pcb_api_clients.py
"""
API Client-Klassen für PCB-Hersteller
Ermöglicht automatische Preisabfragen bei verschiedenen PCB-Herstellern.

Unterstützt:
- JLCPCB
- PCBWay
- Eurocircuits

Jeder Client kann sich authentifizieren, Preisanfragen stellen und Ergebnisse verarbeiten.
"""

import os
import json
import logging
import requests
import time
import hmac
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from abc import ABC, abstractmethod

# Konfiguration für das Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pcb_api_clients")

# Konfigurationsverzeichnis für API-Einstellungen
CONFIG_DIR = Path.home() / ".config" / "pcb_export"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

class BasePCBClient(ABC):
    """Basis-Klasse für PCB-Hersteller API-Clients."""
    
    def __init__(self, name: str, api_base_url: str):
        """
        Initialisiert den Basis-Client.
        
        Args:
            name: Name des PCB-Herstellers
            api_base_url: Basis-URL für API-Anfragen
        """
        self.name = name
        self.api_base_url = api_base_url
        self.auth_token = None
        self.config_file = CONFIG_DIR / f"{name.lower()}_config.json"
        self.session = requests.Session()
        self.last_response = None
        self._load_config()
    
    def _load_config(self) -> Dict:
        """Lädt Konfiguration aus der Datei."""
        config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if 'auth_token' in config:
                        self.auth_token = config['auth_token']
                logger.info(f"{self.name} Konfiguration geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der {self.name} Konfiguration: {e}")
        return config
    
    def _save_config(self, config: Dict) -> bool:
        """Speichert Konfiguration in der Datei."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"{self.name} Konfiguration gespeichert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der {self.name} Konfiguration: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Prüft, ob der Client authentifiziert ist."""
        return self.auth_token is not None
    
    @abstractmethod
    def authenticate(self, **credentials) -> bool:
        """
        Authentifiziert den Client beim PCB-Hersteller.
        Muss von jeder konkreten Implementierung überschrieben werden.
        
        Args:
            **credentials: Herstellerspezifische Anmeldedaten
            
        Returns:
            True bei erfolgreicher Authentifizierung, sonst False
        """
        pass
    
    @abstractmethod
    def get_quote(self, pcb_params: Dict) -> Dict:
        """
        Holt ein Angebot vom PCB-Hersteller.
        Muss von jeder konkreten Implementierung überschrieben werden.
        
        Args:
            pcb_params: Parameter für das PCB
            
        Returns:
            Dictionary mit Angebotsinformationen
        """
        pass
    
    def handle_response(self, response: requests.Response) -> Dict:
        """
        Verarbeitet die API-Antwort und prüft auf Fehler.
        
        Args:
            response: Die API-Antwort
            
        Returns:
            Dictionary mit der verarbeiteten Antwort
        """
        self.last_response = response
        
        try:
            if response.status_code >= 400:
                logger.error(f"{self.name} API Fehler: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP Fehler {response.status_code}",
                    "details": response.text
                }
            
            data = response.json()
            return data
            
        except ValueError as e:
            logger.error(f"{self.name} API Antwort konnte nicht als JSON verarbeitet werden: {e}")
            return {
                "success": False,
                "error": "Ungültiges JSON-Format",
                "details": response.text
            }

    def convert_currency(self, amount: float, from_currency: str, to_currency: str = "EUR") -> float:
        """
        Konvertiert Währungsbeträge (vereinfachte Implementation).
        
        Args:
            amount: Geldbetrag
            from_currency: Quellwährung
            to_currency: Zielwährung (Standard: EUR)
            
        Returns:
            Konvertierter Betrag
        """
        # Vereinfachte Umrechnungsfaktoren (in der Praxis würde eine Währungs-API verwendet)
        conversion_rates = {
            "USD_EUR": 0.91,
            "CNY_EUR": 0.13,
            "EUR_EUR": 1.0,
            "GBP_EUR": 1.17,
            # Bei Bedarf weitere Kurse hinzufügen
        }
        
        conversion_key = f"{from_currency}_{to_currency}"
        if conversion_key in conversion_rates:
            return amount * conversion_rates[conversion_key]
        else:
            logger.warning(f"Währungsumrechnung nicht verfügbar für {conversion_key}")
            return amount


class JLCPCBClient(BasePCBClient):
    """API Client für JLCPCB."""
    
    def __init__(self):
        """Initialisiert den JLCPCB-Client."""
        super().__init__("JLCPCB", "https://api.jlcpcb.com/")
    
    def authenticate(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> bool:
        """
        Authentifiziert mit JLCPCB API.
        
        Args:
            api_key: JLCPCB API Key (optional, kann aus Konfiguration geladen werden)
            api_secret: JLCPCB API Secret (optional, kann aus Konfiguration geladen werden)
            
        Returns:
            True bei erfolgreicher Authentifizierung, sonst False
        """
        # Lade bestehende Konfiguration
        config = self._load_config()
        
        # Verwende Parameter oder Werte aus der Konfiguration
        api_key = api_key or config.get('api_key')
        api_secret = api_secret or config.get('api_secret')
        
        if not api_key or not api_secret:
            logger.error("JLCPCB API Key und Secret werden benötigt")
            return False
        
        # JLCPCB verwendet oft HMAC-basierte Authentifizierung
        timestamp = str(int(time.time()))
        message = f"{api_key}:{timestamp}"
        signature = hmac.new(
            api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        auth_data = {
            "apiKey": api_key,
            "timestamp": timestamp,
            "signature": signature
        }
        
        try:
            response = self.session.post(
                f"{self.api_base_url}auth/token",
                json=auth_data
            )
            
            result = self.handle_response(response)
            
            if result.get("success", False):
                self.auth_token = result.get("data", {}).get("token")
                
                # Speichere Konfiguration
                config.update({
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'auth_token': self.auth_token
                })
                self._save_config(config)
                
                logger.info("JLCPCB Authentifizierung erfolgreich")
                return True
            else:
                logger.error(f"JLCPCB Authentifizierung fehlgeschlagen: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Fehler bei der JLCPCB Authentifizierung: {e}")
            return False
    
    def get_quote(self, pcb_params: Dict) -> Dict:
        """
        Holt ein Angebot von JLCPCB.
        
        Args:
            pcb_params: Parameter für das PCB, sollte folgende Schlüssel enthalten:
                - width_mm: Breite in mm
                - height_mm: Höhe in mm
                - layers: Anzahl der Lagen
                - quantity: Stückzahl
                - pcb_thickness: Dicke in mm
                - copper_weight: Kupfergewicht in oz
                - solder_mask_color: Farbe der Lötstoppmaske
                - silkscreen_color: Farbe des Bestückungsdrucks
                
        Returns:
            Dictionary mit Angebotsinformationen
        """
        if not self.is_authenticated() and not self.authenticate():
            return {
                "success": False,
                "error": "Nicht authentifiziert",
                "details": "Bitte authentifizieren Sie sich zuerst mit API Key und Secret"
            }
        
        # Extrahiere Parameter mit Standardwerten
        width_mm = pcb_params.get('width_mm', 100)
        height_mm = pcb_params.get('height_mm', 100)
        layers = pcb_params.get('layers', 2)
        quantity = pcb_params.get('quantity', 5)
        pcb_thickness = pcb_params.get('pcb_thickness', 1.6)
        copper_weight = pcb_params.get('copper_weight', 1)  # 1oz
        solder_mask_color = pcb_params.get('solder_mask_color', 'Green')
        silkscreen_color = pcb_params.get('silkscreen_color', 'White')
        
        # Berechne Fläche in mm²
        area = width_mm * height_mm
        
        # Formatiere Anfrage nach JLCPCB API-Spezifikation
        quote_data = {
            "pcbParameters": {
                "type": "PCB",
                "material": "FR-4",
                "layers": layers,
                "width": width_mm,
                "height": height_mm,
                "quantity": quantity,
                "thickness": pcb_thickness,
                "outerCopperWeight": copper_weight,
                "solderMaskColor": solder_mask_color,
                "silkscreenColor": silkscreen_color,
                "surfaceFinish": "HASL(with lead)",
                "minHoleDiameter": 0.3,
                "minTrackWidth": 0.127,
                "minSpaceBetweenTracks": 0.127
            }
        }
        
        # Füge optionale Assembly-Parameter hinzu, falls vorhanden
        if 'assembly' in pcb_params and pcb_params['assembly']:
            quote_data["assemblyParameters"] = {
                "side": pcb_params.get('assembly_side', 'top'),
                "uniqueComponentCount": pcb_params.get('unique_components', 0),
                "totalComponentCount": pcb_params.get('total_components', 0)
            }
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(
                f"{self.api_base_url}quote",
                json=quote_data,
                headers=headers
            )
            
            result = self.handle_response(response)
            
            if result.get("success", False):
                quote_info = result.get("data", {})
                
                # Formatiere das Ergebnis in ein einheitliches Format
                return {
                    "success": True,
                    "manufacturer": self.name,
                    "quote_id": quote_info.get("quoteId", ""),
                    "pcb_price": {
                        "amount": quote_info.get("pcbPrice", 0),
                        "currency": "USD"  # JLCPCB verwendet USD
                    },
                    "assembly_price": {
                        "amount": quote_info.get("assemblyPrice", 0),
                        "currency": "USD"
                    },
                    "shipping_price": {
                        "amount": quote_info.get("shippingPrice", 0),
                        "currency": "USD"
                    },
                    "total_price": {
                        "amount": quote_info.get("totalPrice", 0),
                        "currency": "USD"
                    },
                    "estimated_days": quote_info.get("leadTime", 0),
                    "quote_url": quote_info.get("quoteUrl", ""),
                    "raw_response": quote_info
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unbekannter Fehler"),
                    "details": result.get("details", "")
                }
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des JLCPCB-Angebots: {e}")
            return {
                "success": False,
                "error": "API-Anfragefehler",
                "details": str(e)
            }


class PCBWayClient(BasePCBClient):
    """API Client für PCBWay."""
    
    def __init__(self):
        """Initialisiert den PCBWay-Client."""
        super().__init__("PCBWay", "https://api.pcbway.com/")
    
    def authenticate(self, api_key: Optional[str] = None) -> bool:
        """
        Authentifiziert mit PCBWay API.
        
        Args:
            api_key: PCBWay API Key (optional, kann aus Konfiguration geladen werden)
            
        Returns:
            True bei erfolgreicher Authentifizierung, sonst False
        """
        # Lade bestehende Konfiguration
        config = self._load_config()
        
        # Verwende Parameter oder Wert aus der Konfiguration
        api_key = api_key or config.get('api_key')
        
        if not api_key:
            logger.error("PCBWay API Key wird benötigt")
            return False
        
        # PCBWay verwendet einen einfacheren API-Key-basierten Ansatz
        try:
            response = self.session.post(
                f"{self.api_base_url}auth/validate",
                json={"apiKey": api_key}
            )
            
            result = self.handle_response(response)
            
            if result.get("success", False):
                self.auth_token = api_key  # Bei PCBWay ist der API-Key oft auch der Auth-Token
                
                # Speichere Konfiguration
                config.update({
                    'api_key': api_key,
                    'auth_token': self.auth_token
                })
                self._save_config(config)
                
                logger.info("PCBWay Authentifizierung erfolgreich")
                return True
            else:
                logger.error(f"PCBWay Authentifizierung fehlgeschlagen: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Fehler bei der PCBWay Authentifizierung: {e}")
            return False
    
    def get_quote(self, pcb_params: Dict) -> Dict:
        """
        Holt ein Angebot von PCBWay.
        
        Args:
            pcb_params: Parameter für das PCB, sollte folgende Schlüssel enthalten:
                - width_mm: Breite in mm
                - height_mm: Höhe in mm
                - layers: Anzahl der Lagen
                - quantity: Stückzahl
                - pcb_thickness: Dicke in mm
                - copper_weight: Kupfergewicht in oz
                - solder_mask_color: Farbe der Lötstoppmaske
                - silkscreen_color: Farbe des Bestückungsdrucks
                
        Returns:
            Dictionary mit Angebotsinformationen
        """
        if not self.is_authenticated() and not self.authenticate():
            return {
                "success": False,
                "error": "Nicht authentifiziert",
                "details": "Bitte authentifizieren Sie sich zuerst mit API Key"
            }
        
        # Extrahiere Parameter mit Standardwerten
        width_mm = pcb_params.get('width_mm', 100)
        height_mm = pcb_params.get('height_mm', 100)
        layers = pcb_params.get('layers', 2)
        quantity = pcb_params.get('quantity', 5)
        pcb_thickness = pcb_params.get('pcb_thickness', 1.6)
        copper_weight = pcb_params.get('copper_weight', 1)  # 1oz
        solder_mask_color = pcb_params.get('solder_mask_color', 'Green')
        silkscreen_color = pcb_params.get('silkscreen_color', 'White')
        
        # Berechne Fläche in mm²
        area = width_mm * height_mm
        
        # Formatiere Anfrage nach PCBWay API-Spezifikation
        quote_data = {
            "apiKey": self.auth_token,
            "pcbSpec": {
                "boardType": "PCB",
                "boardOutline": {
                    "width": width_mm,
                    "height": height_mm,
                    "units": "mm"
                },
                "quantity": quantity,
                "layers": layers,
                "material": "FR4",
                "thickness": pcb_thickness,
                "minTrackWidth": 5,  # 5 mil
                "minHoleDiameter": 0.3,
                "solderMask": solder_mask_color,
                "silkscreen": silkscreen_color,
                "surfaceFinish": "HASL lead free",
                "copperWeight": copper_weight,
                "viaProcess": "Tenting vias",
                "customerId": ""  # Optional
            }
        }
        
        # Füge optionale Assembly-Parameter hinzu, falls vorhanden
        if 'assembly' in pcb_params and pcb_params['assembly']:
            quote_data["assemblySpec"] = {
                "assemblySides": pcb_params.get('assembly_side', 'Top'),
                "uniquePartCount": pcb_params.get('unique_components', 0),
                "totalSMTParts": pcb_params.get('total_components', 0),
                "throughHoleParts": pcb_params.get('tht_components', 0)
            }
        
        try:
            response = self.session.post(
                f"{self.api_base_url}quote",
                json=quote_data
            )
            
            result = self.handle_response(response)
            
            if result.get("success", False):
                quote_info = result.get("data", {})
                
                # Formatiere das Ergebnis in ein einheitliches Format
                return {
                    "success": True,
                    "manufacturer": self.name,
                    "quote_id": quote_info.get("quoteId", ""),
                    "pcb_price": {
                        "amount": quote_info.get("pcbPrice", 0),
                        "currency": "USD"  # PCBWay verwendet USD
                    },
                    "assembly_price": {
                        "amount": quote_info.get("assemblyPrice", 0),
                        "currency": "USD"
                    },
                    "shipping_price": {
                        "amount": quote_info.get("shippingCost", 0),
                        "currency": "USD"
                    },
                    "total_price": {
                        "amount": quote_info.get("totalPrice", 0),
                        "currency": "USD"
                    },
                    "estimated_days": quote_info.get("productionTime", 0),
                    "quote_url": quote_info.get("quoteUrl", ""),
                    "raw_response": quote_info
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unbekannter Fehler"),
                    "details": result.get("details", "")
                }
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des PCBWay-Angebots: {e}")
            return {
                "success": False,
                "error": "API-Anfragefehler",
                "details": str(e)
            }


class EurocircuitsClient(BasePCBClient):
    """API Client für Eurocircuits."""
    
    def __init__(self):
        """Initialisiert den Eurocircuits-Client."""
        super().__init__("Eurocircuits", "https://api.eurocircuits.com/")
    
    def authenticate(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Authentifiziert mit Eurocircuits API.
        
        Args:
            username: Eurocircuits Benutzername (optional, kann aus Konfiguration geladen werden)
            password: Eurocircuits Passwort (optional, kann aus Konfiguration geladen werden)
            
        Returns:
            True bei erfolgreicher Authentifizierung, sonst False
        """
        # Lade bestehende Konfiguration
        config = self._load_config()
        
        # Verwende Parameter oder Werte aus der Konfiguration
        username = username or config.get('username')
        password = password or config.get('password')
        
        if not username or not password:
            logger.error("Eurocircuits Benutzername und Passwort werden benötigt")
            return False
        
        # Eurocircuits verwendet oft OAuth2 oder ähnliche Authentifizierung
        auth_data = {
            "username": username,
            "password": password,
            "grant_type": "password"
        }
        
        try:
            response = self.session.post(
                f"{self.api_base_url}auth/token",
                data=auth_data
            )
            
            result = self.handle_response(response)
            
            if result.get("success", False) or "access_token" in result:
                self.auth_token = result.get("access_token")
                
                # Speichere Konfiguration
                config.update({
                    'username': username,
                    'password': password,
                    'auth_token': self.auth_token
                })
                self._save_config(config)
                
                logger.info("Eurocircuits Authentifizierung erfolgreich")
                return True
            else:
                logger.error(f"Eurocircuits Authentifizierung fehlgeschlagen: {result.get('error_description', result.get('error'))}")
                return False
                
        except Exception as e:
            logger.error(f"Fehler bei der Eurocircuits Authentifizierung: {e}")
            return False
    
    def get_quote(self, pcb_params: Dict) -> Dict:
        """
        Holt ein Angebot von Eurocircuits.
        
        Args:
            pcb_params: Parameter für das PCB, sollte folgende Schlüssel enthalten:
                - width_mm: Breite in mm
                - height_mm: Höhe in mm
                - layers: Anzahl der Lagen
                - quantity: Stückzahl
                - pcb_thickness: Dicke in mm
                - copper_weight: Kupfergewicht in oz
                - solder_mask_color: Farbe der Lötstoppmaske
                - silkscreen_color: Farbe des Bestückungsdrucks
                
        Returns:
            Dictionary mit Angebotsinformationen
        """
        if not self.is_authenticated() and not self.authenticate():
            return {
                "success": False,
                "error": "Nicht authentifiziert",
                "details": "Bitte authentifizieren Sie sich zuerst mit Benutzername und Passwort"
            }
        
        # Extrahiere Parameter mit Standardwerten
        width_mm = pcb_params.get('width_mm', 100)
        height_mm = pcb_params.get('height_mm', 100)
        layers = pcb_params.get('layers', 2)
        quantity = pcb_params.get('quantity', 5)
        pcb_thickness = pcb_params.get('pcb_thickness', 1.6)
        copper_weight = pcb_params.get('copper_weight', 1)  # 1oz
        solder_mask_color = pcb_params.get('solder_mask_color', 'Green')
        silkscreen_color = pcb_params.get('silkscreen_color', 'White')
        
        # Konvertiere Copper Weight von oz zu µm für Eurocircuits (Europa verwendet µm)
        copper_thickness_um = copper_weight * 35  # 1 oz ≈ 35 µm
        
        # Formatiere Anfrage nach Eurocircuits API-Spezifikation
        quote_data = {
            "technology": "PCB PROTO",  # Oder "PCB POOL" für Serienproduktion
            "dimensions": {
                "width": width_mm,
                "height": height_mm
            },
            "quantity": quantity,
            "layers": layers,
            "material": "FR4",
            "thickness": pcb_thickness,
            "solderMask": solder_mask_color,
            "silkScreen": silkscreen_color,
            "surfaceFinish": "Chemical Ni/Au (ENIG)",
            "copperThickness": copper_thickness_um,
            "minTrackWidth": 150,  # 150 µm
            "minDrillSize": 0.3  # 0.3 mm
        }
        
        # Füge optionale Assembly-Parameter hinzu, falls vorhanden
        if 'assembly' in pcb_params and pcb_params['assembly']:
            quote_data["assembly"] = {
                "sides": pcb_params.get('assembly_side', 'Top'),
                "uniqueComponents": pcb_params.get('unique_components', 0),
                "totalPlacingPoints": pcb_params.get('total_components', 0),
                "throughHoleComponents": pcb_params.get('tht_components', 0)
            }
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = self.session.post(
                f"{self.api_base_url}quote/calculate",
                json=quote_data,
                headers=headers
            )
            
            result = self.handle_response(response)
            
            if result.get("success", False) or "quotation" in result:
                quote_info = result.get("quotation", {})
                
                # Formatiere das Ergebnis in ein einheitliches Format
                return {
                    "success": True,
                    "manufacturer": self.name,
                    "quote_id": quote_info.get("quotationId", ""),
                    "pcb_price": {
                        "amount": quote_info.get("pcbPrice", 0),
                        "currency": "EUR"  # Eurocircuits verwendet EUR
                    },
                    "assembly_price": {
                        "amount": quote_info.get("assemblyPrice", 0),
                        "currency": "EUR"
                    },
                    "shipping_price": {
                        "amount": quote_info.get("shippingCost", 0),
                        "currency": "EUR"
                    },
                    "total_price": {
                        "amount": quote_info.get("totalPrice", 0),
                        "currency": "EUR"
                    },
                    "estimated_days": quote_info.get("deliveryTime", 0),
                    "quote_url": quote_info.get("quotationUrl", ""),
                    "raw_response": quote_info
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unbekannter Fehler"),
                    "details": result.get("details", "")
                }
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Eurocircuits-Angebots: {e}")
            return {
                "success": False,
                "error": "API-Anfragefehler",
                "details": str(e)
            }


# Funktion zum Instanziieren des richtigen Client-Objekts basierend auf dem Herstellernamen
def get_pcb_client(manufacturer: str) -> Optional[BasePCBClient]:
    """
    Gibt eine Instanz des passenden PCB API-Clients zurück.
    
    Args:
        manufacturer: Name des PCB-Herstellers (kleingeschrieben)
        
    Returns:
        Instanz des API-Clients oder None, wenn nicht unterstützt
    """
    clients = {
        "jlcpcb": JLCPCBClient,
        "pcbway": PCBWayClient,
        "eurocircuits": EurocircuitsClient
    }
    
    if manufacturer.lower() in clients:
        return clients[manufacturer.lower()]()
    else:
        logger.error(f"Kein API-Client verfügbar für '{manufacturer}'")
        return None


# Funktion zur Währungsumrechnung (für Vergleiche zwischen Herstellern)
def format_price(amount: float, currency: str = "EUR", decimal_places: int = 2) -> str:
    """
    Formatiert einen Preis mit Währungssymbol.
    
    Args:
        amount: Geldbetrag
        currency: Währung (Standard: EUR)
        decimal_places: Anzahl der Dezimalstellen
        
    Returns:
        Formatierter Preis als String
    """
    currency_symbols = {
        "EUR": "€",
        "USD": "$",
        "CNY": "¥",
        "GBP": "£"
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    if currency in ["EUR", "GBP"]:
        # Europäische Notation mit Symbol nach dem Betrag
        return f"{amount:.{decimal_places}f} {symbol}"
    else:
        # US/Internationale Notation mit Symbol vor dem Betrag
        return f"{symbol}{amount:.{decimal_places}f}"


# Hilfsfunktion zum Vergleichen von Angeboten
def compare_quotes(quotes: List[Dict]) -> Dict:
    """
    Vergleicht Angebote verschiedener Hersteller und gibt eine Zusammenfassung zurück.
    
    Args:
        quotes: Liste von Angebotsdictionaries
        
    Returns:
        Dictionary mit Vergleichsinformationen
    """
    # Konvertiere alle Preise zu EUR für Vergleichbarkeit
    comparison = {
        "cheapest": None,
        "fastest": None,
        "quotes": []
    }
    
    for quote in quotes:
        if not quote.get("success", False):
            continue
        
        # Erstelle ein vergleichbares Quote-Objekt
        normalized_quote = {
            "manufacturer": quote["manufacturer"],
            "total_price_original": {
                "amount": quote["total_price"]["amount"],
                "currency": quote["total_price"]["currency"],
                "formatted": format_price(quote["total_price"]["amount"], quote["total_price"]["currency"])
            },
            "total_price_eur": 0.0,
            "estimated_days": quote.get("estimated_days", 0),
            "quote_url": quote.get("quote_url", "")
        }
        
        # Konvertiere zu EUR für Vergleichbarkeit (falls nicht schon in EUR)
        client = get_pcb_client(quote["manufacturer"].lower())
        if client:
            amount_eur = client.convert_currency(
                quote["total_price"]["amount"],
                quote["total_price"]["currency"],
                "EUR"
            )
            normalized_quote["total_price_eur"] = amount_eur
            normalized_quote["total_price_eur_formatted"] = format_price(amount_eur, "EUR")
        
        comparison["quotes"].append(normalized_quote)
    
    # Finde günstigstes und schnellstes Angebot
    if comparison["quotes"]:
        # Sortiere nach Preis (aufsteigend)
        sorted_by_price = sorted(comparison["quotes"], key=lambda x: x["total_price_eur"])
        comparison["cheapest"] = sorted_by_price[0]
        
        # Sortiere nach Lieferzeit (aufsteigend)
        sorted_by_time = sorted(comparison["quotes"], key=lambda x: x["estimated_days"])
        comparison["fastest"] = sorted_by_time[0]
    
    return comparison
