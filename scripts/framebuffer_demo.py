#!/usr/bin/env python3
"""
Demonstriert die Auswirkung verschiedener Framebuffer-Konfigurationen auf den RAM-Verbrauch.
Dieses Skript zeigt die EMU-01 Framebuilder-Korrektur in Aktion.
"""

import sys
import os
import time
import argparse
from tabulate import tabulate

# Füge Importpfad für die Emulator-Module hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.emulation.frame_buffer import FrameBuffer, PixelFormat
    from src.emulation.emulator import RP2040Emulator, CameraEmulator, AdaptiveMode
except ImportError:
    print("Fehler beim Importieren der Emulator-Module. Bitte stellen Sie sicher, dass Sie im richtigen Verzeichnis sind.")
    sys.exit(1)

def print_header(title):
    """Gibt eine formatierte Überschrift aus."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_memory_stats(emulator, additional_info=""):
    """Gibt detaillierte Speicherstatistiken für den Emulator aus."""
    ram_usage = emulator.get_ram_usage()
    framebuffer_ram = emulator.framebuffer_ram_bytes
    system_ram = emulator.system_ram_overhead
    model_ram = emulator.ram_used if emulator.firmware_loaded else 0
    
    total_ram_kb = ram_usage / 1024
    free_ram_kb = (emulator.ram_size_bytes - ram_usage) / 1024
    framebuffer_kb = framebuffer_ram / 1024
    system_kb = system_ram / 1024
    model_kb = model_ram / 1024
    
    usage_percent = (ram_usage / emulator.ram_size_bytes) * 100
    
    stats = [
        ["Gesamt-RAM", f"{total_ram_kb:.1f} KB", f"{usage_percent:.1f}%"],
        ["Freier RAM", f"{free_ram_kb:.1f} KB", f"{100-usage_percent:.1f}%"],
        ["Framebuffer", f"{framebuffer_kb:.1f} KB", f"{(framebuffer_ram/emulator.ram_size_bytes)*100:.1f}%"],
        ["System-Overhead", f"{system_kb:.1f} KB", f"{(system_ram/emulator.ram_size_bytes)*100:.1f}%"],
    ]
    
    if emulator.firmware_loaded:
        stats.append(["Modell (Tensor Arena)", f"{model_kb:.1f} KB", f"{(model_ram/emulator.ram_size_bytes)*100:.1f}%"])
    
    if additional_info:
        print(f"\n{additional_info}\n")
        
    print(tabulate(stats, headers=["Komponente", "Größe", "Anteil"], tablefmt="grid"))
    
    fb_stats = emulator.camera.get_frame_buffer_stats()
    
    print(f"\nFramebuffer-Details:")
    print(f"  Format: {fb_stats['pixel_format']}")
    print(f"  Auflösung: {emulator.camera.width}x{emulator.camera.height}")
    print(f"  Größe: {fb_stats['total_size_kb']:.1f} KB")
    
    # RAM-Statusbalken
    bar_width = 50
    used_width = int((ram_usage / emulator.ram_size_bytes) * bar_width)
    framebuffer_width = int((framebuffer_ram / emulator.ram_size_bytes) * bar_width)
    system_width = int((system_ram / emulator.ram_size_bytes) * bar_width)
    model_width = int((model_ram / emulator.ram_size_bytes) * bar_width)
    
    print("\nRAM-Auslastung:")
    print(f"[{'M'*model_width}{'S'*system_width}{'F'*framebuffer_width}{' '*(bar_width-used_width)}] {usage_percent:.1f}%")
    print(f" M = Modell, S = System, F = Framebuffer")

def demo_format_impact(emulator):
    """Demonstriert den Einfluss verschiedener Pixelformate auf den RAM-Verbrauch."""
    print_header("Einfluss des Pixelformats auf den RAM-Verbrauch")
    
    formats = [
        (PixelFormat.RGB888, "RGB888 (3 Bytes/Pixel)"),
        (PixelFormat.RGB565, "RGB565 (2 Bytes/Pixel)"),
        (PixelFormat.GRAYSCALE, "Graustufen (1 Byte/Pixel)"),
        (PixelFormat.YUV422, "YUV422 (2 Bytes/Pixel)")
    ]
    
    for pixel_format, description in formats:
        emulator.set_camera_pixel_format(pixel_format)
        print_memory_stats(emulator, f"Pixelformat: {description}")
        time.sleep(1)  # Kurze Pause für bessere Lesbarkeit

def demo_resolution_impact(emulator):
    """Demonstriert den Einfluss verschiedener Auflösungen auf den RAM-Verbrauch."""
    print_header("Einfluss der Auflösung auf den RAM-Verbrauch")
    
    resolutions = [
        (160, 120, "QQVGA (160x120)"),
        (320, 240, "QVGA (320x240)"),
        (640, 480, "VGA (640x480)"),
        (800, 600, "SVGA (800x600)")
    ]
    
    # Setze auf RGB565 für realistischere Werte
    emulator.set_camera_pixel_format(PixelFormat.RGB565)
    
    for width, height, description in resolutions:
        # Prüfe, ob diese Auflösung zu einem RAM-Überlauf führen würde
        new_buffer_size = width * height * 2  # RGB565 = 2 Bytes pro Pixel
        total_ram = emulator.system_ram_overhead + new_buffer_size
        if emulator.firmware_loaded:
            total_ram += emulator.ram_used
        
        if total_ram > emulator.ram_size_bytes:
            print(f"\nWarnung: Auflösung {description} würde zu RAM-Überlauf führen!")
            print(f"Benötigter RAM: {total_ram/1024:.1f}KB, Verfügbar: {emulator.ram_size_bytes/1024:.1f}KB")
            continue
            
        emulator.set_camera_format(width, height, PixelFormat.RGB565)
        print_memory_stats(emulator, f"Auflösung: {description}")
        time.sleep(1)  # Kurze Pause für bessere Lesbarkeit

def demo_firmware_loading(emulator):
    """Demonstriert die Auswirkung der Firmware-Größe auf den RAM-Verbrauch."""
    print_header("Auswirkung der Firmware-Größe auf den RAM-Verbrauch")
    
    # Setze auf standardmäßige Konfiguration zurück
    emulator.set_camera_format(320, 240, PixelFormat.RGB565)
    
    # Zeige aktuellen Zustand ohne Firmware
    print_memory_stats(emulator, "Ohne Firmware")
    
    # Berechne verfügbaren RAM für Firmware (abzüglich System-Overhead und Framebuffer)
    available_ram = emulator.ram_size_bytes - emulator.system_ram_overhead - emulator.framebuffer_ram_bytes
    
    firmware_sizes = [
        (int(available_ram * 0.25), "Kleine Firmware (25% des verfügbaren RAMs)"),
        (int(available_ram * 0.5), "Mittlere Firmware (50% des verfügbaren RAMs)"),
        (int(available_ram * 0.75), "Große Firmware (75% des verfügbaren RAMs)"),
        (int(available_ram * 0.95), "Sehr große Firmware (95% des verfügbaren RAMs)"),
        (int(available_ram * 1.1), "Zu große Firmware (110% des verfügbaren RAMs)")
    ]
    
    for ram_usage, description in firmware_sizes:
        firmware = {
            'ram_usage_bytes': ram_usage,
            'total_size_bytes': ram_usage * 2  # Annahme: Flash-Verbrauch ist doppelt so groß wie RAM
        }
        
        print(f"\n{description}")
        print(f"RAM-Bedarf: {ram_usage/1024:.1f}KB, Flash: {firmware['total_size_bytes']/1024:.1f}KB")
        
        try:
            emulator.load_firmware(firmware)
            print("Firmware erfolgreich geladen.")
            print_memory_stats(emulator)
        except Exception as e:
            print(f"Fehler beim Laden der Firmware: {e}")
            
        time.sleep(1)  # Kurze Pause für bessere Lesbarkeit

def demo_before_after_correction():
    """Demonstriert den Unterschied zwischen dem alten und dem korrigierten Emulator."""
    print_header("Vergleich: Vor und nach der EMU-01 Framebuilder-Korrektur")
    
    # Simuliere den alten Emulator (ohne Berücksichtigung des Framebuffers)
    class OldEmulator:
        def __init__(self):
            self.ram_size_bytes = 264 * 1024
            self.system_ram_overhead = 40 * 1024
            
        def check_firmware(self, firmware_ram):
            total_ram = firmware_ram + self.system_ram_overhead
            return total_ram <= self.ram_size_bytes
    
    old_emulator = OldEmulator()
    
    # Erstelle einen neuen korrigierten Emulator
    new_emulator = RP2040Emulator()
    new_emulator.set_camera_format(320, 240, PixelFormat.RGB565)  # Realistisches Format
    
    # Typischer RAM-Bedarf für ein Modell
    model_ram = 90 * 1024  # 90KB Tensor Arena
    
    # Alte Berechnung
    old_total_ram = model_ram + old_emulator.system_ram_overhead
    old_ram_free = old_emulator.ram_size_bytes - old_total_ram
    old_ram_percent = (old_total_ram / old_emulator.ram_size_bytes) * 100
    
    # Neue Berechnung
    new_total_ram = model_ram + new_emulator.system_ram_overhead + new_emulator.framebuffer_ram_bytes
    new_ram_free = new_emulator.ram_size_bytes - new_total_ram
    new_ram_percent = (new_total_ram / new_emulator.ram_size_bytes) * 100
    
    # Ausgabe der Ergebnisse
    comparison = [
        ["Vor Korrektur", f"{old_total_ram/1024:.1f}KB", f"{old_ram_free/1024:.1f}KB", f"{old_ram_percent:.1f}%"],
        ["Nach Korrektur", f"{new_total_ram/1024:.1f}KB", f"{new_ram_free/1024:.1f}KB", f"{new_ram_percent:.1f}%"]
    ]
    
    print(tabulate(comparison, 
                 headers=["Emulator-Version", "Benötigter RAM", "Freier RAM", "RAM-Auslastung"], 
                 tablefmt="grid"))
    
    print("\nAufschlüsselung des korrigierten RAM-Verbrauchs:")
    print(f"Modell (Tensor Arena): {model_ram/1024:.1f}KB")
    print(f"System-Overhead: {new_emulator.system_ram_overhead/1024:.1f}KB")
    print(f"Framebuffer ({new_emulator.camera.frame_buffer.pixel_format.name}): "
          f"{new_emulator.framebuffer_ram_bytes/1024:.1f}KB")
    
    # Zeige Warnung, wenn die alte Berechnung zu unzuverlässigen Ergebnissen führt
    if old_ram_percent < 60 and new_ram_percent > 85:
        print("\nWARNUNG: Die alte Emulator-Version unterschätzt den RAM-Bedarf erheblich!")
        print("Der tatsächliche RAM-Verbrauch liegt deutlich höher und könnte zu Laufzeitproblemen führen.")
    
    if new_ram_percent > 95:
        print("\nKRITISCH: Die korrigierte Berechnung zeigt, dass der RAM nahezu erschöpft ist!")
        print("Dies kann zu unvorhersehbarem Verhalten oder Abstürzen auf der tatsächlichen Hardware führen.")

def main():
    parser = argparse.ArgumentParser(description="Demonstriert die EMU-01 Framebuilder-Korrektur")
    parser.add_argument("--all", action="store_true", help="Alle Demos ausführen")
    parser.add_argument("--format", action="store_true", help="Einfluss des Pixelformats demonstrieren")
    parser.add_argument("--resolution", action="store_true", help="Einfluss der Auflösung demonstrieren")
    parser.add_argument("--firmware", action="store_true", help="Auswirkung der Firmware-Größe demonstrieren")
    parser.add_argument("--compare", action="store_true", help="Vergleich vor/nach der Korrektur")
    
    args = parser.parse_args()
    
    # Führe alle Demos aus, wenn keine spezifische Demo ausgewählt wurde
    if not any([args.format, args.resolution, args.firmware, args.compare]):
        args.all = True
    
    print_header("EMU-01 Framebuilder-Korrektur Demo")
    print("Diese Demo zeigt die Auswirkungen des Kamera-Framebuffers auf den RAM-Verbrauch.")
    
    if args.all or args.compare:
        demo_before_after_correction()
    
    # Erstelle einen Emulator für die anderen Demos
    emulator = RP2040Emulator()
    
    if args.all or args.format:
        demo_format_impact(emulator)
    
    if args.all or args.resolution:
        demo_resolution_impact(emulator)
    
    if args.all or args.firmware:
        demo_firmware_loading(emulator)
    
    print("\nDemo abgeschlossen.")

if __name__ == "__main__":
    main()
