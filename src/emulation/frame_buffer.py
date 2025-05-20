"""
Frame Buffer Simulator für OV2640 Kamera auf RP2040.

Implementiert eine präzise Simulation des Kamera-Framebuffers im RAM,
mit korrekter Speicherausrichtung, Padding und Unterstützung für verschiedene Pixelformate.
"""

import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class PixelFormat(Enum):
    """Unterstützte Pixelformate der OV2640-Kamera."""
    RGB888 = 1  # 3 Bytes pro Pixel (24-bit)
    RGB565 = 2  # 2 Bytes pro Pixel (16-bit)
    GRAYSCALE = 3  # 1 Byte pro Pixel (8-bit)
    YUV422 = 4  # 2 Bytes pro Pixel (16-bit, YUV Format)

class FrameBuffer:
    """
    Präzise Simulation des Kamera-Framebuffers im RAM.
    
    Diese Klasse implementiert eine genaue Darstellung des Framebuffers,
    wie er im RAM des RP2040 existieren würde, mit korrekter Byte-Ausrichtung,
    Padding und Unterstützung für alle relevanten Pixelformate.
    """
    
    def __init__(self, width: int, height: int, pixel_format: PixelFormat = PixelFormat.RGB888):
        """
        Initialisiert einen neuen Framebuffer.
        
        Args:
            width: Breite des Bildes in Pixeln
            height: Höhe des Bildes in Pixeln
            pixel_format: Pixelformat (RGB888, RGB565, GRAYSCALE, YUV422)
        """
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        
        # Byte pro Pixel für jedes Format
        self.bytes_per_pixel = {
            PixelFormat.RGB888: 3,
            PixelFormat.RGB565: 2,
            PixelFormat.GRAYSCALE: 1,
            PixelFormat.YUV422: 2
        }[pixel_format]
        
        # Ausrichtung auf 4-Byte-Grenzen (für optimale Leistung auf ARM-Prozessoren)
        self.row_bytes = self._calculate_row_bytes()
        
        # Gesamtgröße des Framebuffers in Bytes
        self.total_size_bytes = self.row_bytes * self.height
        
        # Erstelle den tatsächlichen Speicherbereich für den Framebuffer
        self.buffer = bytearray(self.total_size_bytes)
        
        # Tracking für Schreib-/Lesezugriffe
        self.last_write_time = 0
        self.total_write_operations = 0
        self.total_read_operations = 0
        self.write_in_progress = False
        self.frame_count = 0
        self.frame_drops = 0
        
        logger.info(
            f"Framebuffer erstellt: {width}x{height} {pixel_format.name}, "
            f"{self.total_size_bytes/1024:.2f}KB, "
            f"{self.row_bytes} Bytes pro Zeile"
        )
    
    def _calculate_row_bytes(self) -> int:
        """
        Berechnet die Anzahl der Bytes pro Zeile mit korrektem Padding.
        
        RP2040/ARM-Prozessoren arbeiten am effizientesten mit Speicher, der auf 4-Byte-Grenzen
        ausgerichtet ist, daher fügen wir Padding hinzu, um sicherzustellen, dass jede Zeile
        ein Vielfaches von 4 Bytes ist.
        
        Returns:
            Anzahl der Bytes pro Zeile inklusive Padding
        """
        raw_row_bytes = self.width * self.bytes_per_pixel
        # Runde auf das nächste Vielfache von 4 auf (für 4-Byte-Ausrichtung)
        return (raw_row_bytes + 3) & ~3
    
    def begin_frame_write(self) -> bool:
        """
        Beginnt einen Schreibvorgang eines neuen Frames.
        
        Returns:
            True, wenn der Schreibvorgang begonnen werden kann, False sonst
        """
        if self.write_in_progress:
            logger.warning("Versuch, einen neuen Frame zu schreiben, während ein anderer noch verarbeitet wird")
            self.frame_drops += 1
            return False
        
        self.write_in_progress = True
        self.last_write_time = time.time()
        return True
    
    def write_pixel_data(self, data: Union[np.ndarray, bytes, bytearray], offset: int = 0, length: int = None) -> int:
        """
        Schreibt Pixeldaten in den Framebuffer.
        
        Args:
            data: Pixeldaten zum Schreiben
            offset: Startposition im Framebuffer
            length: Anzahl der zu schreibenden Bytes (oder None für alle Daten)
            
        Returns:
            Anzahl der geschriebenen Bytes
        """
        if not self.write_in_progress:
            logger.error("Versuch, Pixeldaten zu schreiben ohne begin_frame_write() aufzurufen")
            return 0
        
        # Konvertiere numpy-Array zu Bytes wenn nötig
        if isinstance(data, np.ndarray):
            # Stelle sicher, dass das Array das richtige Format hat
            data_bytes = self._convert_numpy_to_bytes(data)
        else:
            data_bytes = data
        
        if length is None:
            length = len(data_bytes)
        
        # Überprüfe, ob der Schreibvorgang innerhalb der Puffergrenzen liegt
        if offset + length > self.total_size_bytes:
            logger.error(
                f"Speicherüberlauf! Versuch, {length} Bytes an Position {offset} "
                f"zu schreiben (Puffergröße: {self.total_size_bytes})"
            )
            # Begrenze den Schreibvorgang auf die verfügbare Größe
            length = max(0, self.total_size_bytes - offset)
        
        # Schreibe die Daten in den Puffer
        if length > 0:
            self.buffer[offset:offset+length] = data_bytes[:length]
            self.total_write_operations += 1
        
        return length
    
    def end_frame_write(self) -> None:
        """Beendet einen Frameschreibvorgang."""
        if not self.write_in_progress:
            logger.warning("end_frame_write() aufgerufen, obwohl kein Schreibvorgang aktiv war")
            return
        
        self.write_in_progress = False
        self.frame_count += 1
        
        # Berechne Schreibzeit für Debugging
        write_time_ms = (time.time() - self.last_write_time) * 1000
        logger.debug(f"Frame {self.frame_count} geschrieben in {write_time_ms:.2f}ms")
    
    def get_frame_as_numpy(self) -> np.ndarray:
        """
        Liest den aktuellen Frame als numpy-Array.
        
        Returns:
            Numpy-Array mit den Framebuffer-Daten
        """
        self.total_read_operations += 1
        
        # Erstelle ein numpy-Array mit der korrekten Form und Datentyp
        if self.pixel_format == PixelFormat.RGB888:
            # Für RGB888: Reshape zu [Höhe, Breite, 3]
            # Wir müssen das Padding am Ende jeder Zeile berücksichtigen
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                row_start = y * self.row_bytes
                for x in range(self.width):
                    pixel_start = row_start + x * 3
                    frame[y, x, 0] = self.buffer[pixel_start]
                    frame[y, x, 1] = self.buffer[pixel_start + 1]
                    frame[y, x, 2] = self.buffer[pixel_start + 2]
            
        elif self.pixel_format == PixelFormat.RGB565:
            # Für RGB565: Wir müssen die 16-Bit-Werte in RGB-Komponenten umwandeln
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                row_start = y * self.row_bytes
                for x in range(self.width):
                    pixel_start = row_start + x * 2
                    # RGB565: RRRRRGGG GGGBBBBB
                    pixel_value = (self.buffer[pixel_start + 1] << 8) | self.buffer[pixel_start]
                    r = ((pixel_value >> 11) & 0x1F) << 3
                    g = ((pixel_value >> 5) & 0x3F) << 2
                    b = (pixel_value & 0x1F) << 3
                    frame[y, x, 0] = r
                    frame[y, x, 1] = g
                    frame[y, x, 2] = b
            
        elif self.pixel_format == PixelFormat.GRAYSCALE:
            # Für Grauscale: Reshape zu [Höhe, Breite]
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
            for y in range(self.height):
                row_start = y * self.row_bytes
                for x in range(self.width):
                    frame[y, x] = self.buffer[row_start + x]
            
        elif self.pixel_format == PixelFormat.YUV422:
            # Für YUV422: Konvertiere zu RGB
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                row_start = y * self.row_bytes
                for x in range(0, self.width, 2):  # YUV422 hat Y für jeden Pixel, aber U/V geteilt
                    pixel_start = row_start + x * 2
                    
                    # YUV422 Format: [Y0, U, Y1, V]
                    y0 = self.buffer[pixel_start]
                    u = self.buffer[pixel_start + 1]
                    y1 = self.buffer[pixel_start + 2] if x + 1 < self.width else y0
                    v = self.buffer[pixel_start + 3] if x + 1 < self.width else 128
                    
                    # YUV zu RGB Konvertierung
                    frame[y, x] = self._yuv_to_rgb(y0, u, v)
                    if x + 1 < self.width:
                        frame[y, x + 1] = self._yuv_to_rgb(y1, u, v)
        
        return frame
    
    def _convert_numpy_to_bytes(self, array: np.ndarray) -> bytes:
        """
        Konvertiert ein numpy-Array in das korrekte Byteformat für den Framebuffer.
        
        Args:
            array: Eingabe-numpy-Array
            
        Returns:
            Bytes-Objekt mit den konvertierten Daten
        """
        if self.pixel_format == PixelFormat.RGB888:
            # Prüfe, ob das Array die richtige Form hat
            if len(array.shape) == 3 and array.shape[2] == 3:
                # Direktes Kopieren möglich
                return array.tobytes()
            else:
                raise ValueError(f"Ungültige Array-Form für RGB888: {array.shape}")
                
        elif self.pixel_format == PixelFormat.RGB565:
            # Konvertiere RGB zu RGB565
            if len(array.shape) == 3 and array.shape[2] == 3:
                # Erstelle einen Ausgabepuffer mit der richtigen Größe
                output = bytearray(self.width * self.height * 2)
                
                for y in range(min(array.shape[0], self.height)):
                    for x in range(min(array.shape[1], self.width)):
                        r, g, b = array[y, x]
                        # RGB565 Format: RRRRRGGG GGGBBBBB
                        pixel = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                        idx = (y * self.width + x) * 2
                        output[idx] = pixel & 0xFF
                        output[idx + 1] = (pixel >> 8) & 0xFF
                
                return bytes(output)
            else:
                raise ValueError(f"Ungültige Array-Form für RGB565: {array.shape}")
                
        elif self.pixel_format == PixelFormat.GRAYSCALE:
            # Prüfe, ob wir ein RGB-Array haben, das zu Graustufen konvertiert werden muss
            if len(array.shape) == 3 and array.shape[2] == 3:
                # Konvertiere RGB zu Graustufen mit gewichteten Faktoren
                gray = np.dot(array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                return gray.tobytes()
            elif len(array.shape) == 2:
                # Bereits Graustufen
                return array.tobytes()
            else:
                raise ValueError(f"Ungültige Array-Form für Graustufen: {array.shape}")
                
        elif self.pixel_format == PixelFormat.YUV422:
            # Konvertiere RGB zu YUV422
            if len(array.shape) == 3 and array.shape[2] == 3:
                # Erstelle einen Ausgabepuffer mit der richtigen Größe
                output = bytearray(self.width * self.height * 2)
                
                for y in range(min(array.shape[0], self.height)):
                    for x in range(0, min(array.shape[1], self.width), 2):
                        # Verarbeite 2 Pixel auf einmal für YUV422
                        r0, g0, b0 = array[y, x]
                        # Handle Edge Case: Wenn wir am Bildrand sind
                        if x + 1 < min(array.shape[1], self.width):
                            r1, g1, b1 = array[y, x + 1]
                        else:
                            r1, g1, b1 = r0, g0, b0
                        
                        # RGB zu YUV Konvertierung
                        y0, u0, v0 = self._rgb_to_yuv(r0, g0, b0)
                        y1, u1, v1 = self._rgb_to_yuv(r1, g1, b1)
                        
                        # Für YUV422 teilen wir U und V zwischen benachbarten Pixeln
                        u = (u0 + u1) // 2
                        v = (v0 + v1) // 2
                        
                        # Speichere im YUV422-Format: [Y0, U, Y1, V]
                        idx = (y * self.width + x) * 2
                        output[idx] = y0
                        output[idx + 1] = u
                        if x + 1 < min(array.shape[1], self.width):
                            output[idx + 2] = y1
                            output[idx + 3] = v
                
                return bytes(output)
            else:
                raise ValueError(f"Ungültige Array-Form für YUV422: {array.shape}")
        
        raise ValueError(f"Nicht unterstütztes Pixelformat: {self.pixel_format}")
    
    def _rgb_to_yuv(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """
        Konvertiert RGB-Werte zu YUV.
        
        Args:
            r: Rot-Komponente (0-255)
            g: Grün-Komponente (0-255)
            b: Blau-Komponente (0-255)
            
        Returns:
            Tuple mit Y, U, V Werten
        """
        y = int(0.299 * r + 0.587 * g + 0.114 * b)
        u = int(-0.14713 * r - 0.28886 * g + 0.436 * b + 128)
        v = int(0.615 * r - 0.51499 * g - 0.10001 * b + 128)
        
        # Begrenze auf 0-255
        return (
            max(0, min(255, y)),
            max(0, min(255, u)),
            max(0, min(255, v))
        )
    
    def _yuv_to_rgb(self, y: int, u: int, v: int) -> Tuple[int, int, int]:
        """
        Konvertiert YUV-Werte zu RGB.
        
        Args:
            y: Y-Komponente (Helligkeit)
            u: U-Komponente (Blau-Projektion)
            v: V-Komponente (Rot-Projektion)
            
        Returns:
            Tuple mit R, G, B Werten
        """
        c = y - 16
        d = u - 128
        e = v - 128
        
        r = int((298 * c + 409 * e + 128) >> 8)
        g = int((298 * c - 100 * d - 208 * e + 128) >> 8)
        b = int((298 * c + 516 * d + 128) >> 8)
        
        # Begrenze auf 0-255
        return (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b))
        )
    
    def get_statistics(self) -> Dict:
        """
        Liefert Statistiken über den Framebuffer.
        
        Returns:
            Dictionary mit Statistiken
        """
        return {
            'total_size_kb': self.total_size_bytes / 1024,
            'width': self.width,
            'height': self.height,
            'pixel_format': self.pixel_format.name,
            'bytes_per_pixel': self.bytes_per_pixel,
            'row_bytes': self.row_bytes,
            'frames_processed': self.frame_count,
            'frames_dropped': self.frame_drops,
            'write_operations': self.total_write_operations,
            'read_operations': self.total_read_operations
        }
    
    def clear(self) -> None:
        """Löscht den Framebuffer-Inhalt."""
        self.buffer = bytearray(self.total_size_bytes)
    
    def get_memory_layout(self) -> Dict:
        """
        Liefert Details zum Speicherlayout des Framebuffers.
        
        Returns:
            Dictionary mit Informationen zum Speicherlayout
        """
        raw_row_bytes = self.width * self.bytes_per_pixel
        padding_bytes = self.row_bytes - raw_row_bytes
        
        return {
            'total_size_bytes': self.total_size_bytes,
            'raw_row_bytes': raw_row_bytes,
            'aligned_row_bytes': self.row_bytes,
            'padding_bytes_per_row': padding_bytes,
            'total_padding_bytes': padding_bytes * self.height
        }
