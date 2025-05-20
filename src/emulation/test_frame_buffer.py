"""
Tests für die Framebuffer-Implementierung.
"""

import unittest
import numpy as np
from frame_buffer import FrameBuffer, PixelFormat

class TestFrameBuffer(unittest.TestCase):
    """Tests für die Framebuffer-Klasse."""
    
    def test_buffer_initialization(self):
        """Testet die korrekte Initialisierung des Framebuffers."""
        # Test RGB888
        fb_rgb888 = FrameBuffer(320, 240, PixelFormat.RGB888)
        self.assertEqual(fb_rgb888.width, 320)
        self.assertEqual(fb_rgb888.height, 240)
        self.assertEqual(fb_rgb888.pixel_format, PixelFormat.RGB888)
        self.assertEqual(fb_rgb888.bytes_per_pixel, 3)
        # Überprüfe, ob die Zeilenlänge auf 4-Byte-Grenzen ausgerichtet ist
        self.assertEqual(fb_rgb888.row_bytes % 4, 0)
        
        # Test RGB565
        fb_rgb565 = FrameBuffer(320, 240, PixelFormat.RGB565)
        self.assertEqual(fb_rgb565.bytes_per_pixel, 2)
        self.assertEqual(fb_rgb565.row_bytes, 640)  # 320 Pixel * 2 Bytes = 640 (bereits durch 4 teilbar)
        
        # Test GRAYSCALE
        fb_gray = FrameBuffer(320, 240, PixelFormat.GRAYSCALE)
        self.assertEqual(fb_gray.bytes_per_pixel, 1)
        self.assertEqual(fb_gray.row_bytes, 320)  # 320 Pixel * 1 Byte = 320 (bereits durch 4 teilbar)
        
        # Test YUV422
        fb_yuv = FrameBuffer(320, 240, PixelFormat.YUV422)
        self.assertEqual(fb_yuv.bytes_per_pixel, 2)
        self.assertEqual(fb_yuv.row_bytes, 640)  # 320 Pixel * 2 Bytes = 640 (bereits durch 4 teilbar)
    
    def test_buffer_size(self):
        """Testet, ob die Puffergröße korrekt berechnet wird."""
        # RGB888: 320 * 240 * 3 = 230,400, aufgerundet auf 4-Byte-Grenze
        fb_rgb888 = FrameBuffer(320, 240, PixelFormat.RGB888)
        expected_row_bytes = (320 * 3 + 3) & ~3  # Aufrunden auf 4-Byte-Grenze
        expected_size = expected_row_bytes * 240
        self.assertEqual(fb_rgb888.total_size_bytes, expected_size)
        
        # Test mit ungerader Breite, die Padding erfordert
        fb_odd = FrameBuffer(321, 240, PixelFormat.RGB888)
        expected_row_bytes = (321 * 3 + 3) & ~3  # Aufrunden auf 4-Byte-Grenze
        expected_size = expected_row_bytes * 240
        self.assertEqual(fb_odd.total_size_bytes, expected_size)
        
        # Teste, ob die tatsächliche Puffergröße mit der berechneten übereinstimmt
        self.assertEqual(len(fb_odd.buffer), expected_size)
    
    def test_write_read_cycle(self):
        """Testet den Schreib- und Lesezyklus von Pixeldaten."""
        fb = FrameBuffer(4, 4, PixelFormat.RGB888)
        
        # Erzeuge ein Testmuster
        test_pattern = np.zeros((4, 4, 3), dtype=np.uint8)
        test_pattern[0, 0] = [255, 0, 0]    # Rot
        test_pattern[0, 1] = [0, 255, 0]    # Grün
        test_pattern[0, 2] = [0, 0, 255]    # Blau
        test_pattern[0, 3] = [255, 255, 0]  # Gelb
        
        # Schreibe das Testmuster
        fb.begin_frame_write()
        fb.write_pixel_data(test_pattern)
        fb.end_frame_write()
        
        # Lese das Testmuster zurück
        result = fb.get_frame_as_numpy()
        
        # Überprüfe, ob die Daten korrekt sind
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(result[0, 1], [0, 255, 0])
        np.testing.assert_array_equal(result[0, 2], [0, 0, 255])
        np.testing.assert_array_equal(result[0, 3], [255, 255, 0])
    
    def test_rgb565_conversion(self):
        """Testet die RGB565-Konvertierung."""
        fb = FrameBuffer(2, 2, PixelFormat.RGB565)
        
        # Erzeuge ein Testmuster
        test_pattern = np.zeros((2, 2, 3), dtype=np.uint8)
        test_pattern[0, 0] = [255, 0, 0]    # Rot
        test_pattern[0, 1] = [0, 255, 0]    # Grün
        test_pattern[1, 0] = [0, 0, 255]    # Blau
        test_pattern[1, 1] = [255, 255, 0]  # Gelb
        
        # Schreibe das Testmuster
        fb.begin_frame_write()
        fb.write_pixel_data(test_pattern)
        fb.end_frame_write()
        
        # Lese das Testmuster zurück
        result = fb.get_frame_as_numpy()
        
        # Bei RGB565 gibt es Quantisierungsverluste, daher überprüfen wir mit Toleranz
        self.assertTrue(result[0, 0, 0] > 240)  # Rot
        self.assertTrue(result[0, 0, 1] < 16)
        self.assertTrue(result[0, 0, 2] < 16)
        
        self.assertTrue(result[0, 1, 0] < 16)   # Grün
        self.assertTrue(result[0, 1, 1] > 240)
        self.assertTrue(result[0, 1, 2] < 16)
        
        self.assertTrue(result[1, 0, 0] < 16)   # Blau
        self.assertTrue(result[1, 0, 1] < 16)
        self.assertTrue(result[1, 0, 2] > 240)
        
        self.assertTrue(result[1, 1, 0] > 240)  # Gelb
        self.assertTrue(result[1, 1, 1] > 240)
        self.assertTrue(result[1, 1, 2] < 16)
    
    def test_grayscale_conversion(self):
        """Testet die Graustufenkonvertierung."""
        fb = FrameBuffer(2, 2, PixelFormat.GRAYSCALE)
        
        # Erzeuge ein Testmuster
        test_pattern = np.zeros((2, 2, 3), dtype=np.uint8)
        test_pattern[0, 0] = [255, 0, 0]      # Rot (sollte zu mittlerem Grau)
        test_pattern[0, 1] = [0, 255, 0]      # Grün (sollte zu hellem Grau)
        test_pattern[1, 0] = [0, 0, 255]      # Blau (sollte zu dunklem Grau)
        test_pattern[1, 1] = [255, 255, 255]  # Weiß
        
        # Schreibe das Testmuster
        fb.begin_frame_write()
        fb.write_pixel_data(test_pattern)
        fb.end_frame_write()
        
        # Lese das Testmuster zurück
        result = fb.get_frame_as_numpy()
        
        # RGB -> Graustufen-Konvertierung verwendet die Formel:
        # Gray = 0.299*R + 0.587*G + 0.114*B
        expected_0_0 = int(0.299 * 255)
        expected_0_1 = int(0.587 * 255)
        expected_1_0 = int(0.114 * 255)
        expected_1_1 = 255
        
        # Überprüfe mit Toleranz (±1), da Rundungsfehler auftreten können
        self.assertTrue(abs(result[0, 0] - expected_0_0) <= 1)
        self.assertTrue(abs(result[0, 1] - expected_0_1) <= 1)
        self.assertTrue(abs(result[1, 0] - expected_1_0) <= 1)
        self.assertTrue(abs(result[1, 1] - expected_1_1) <= 1)
    
    def test_yuv_conversion(self):
        """Testet die YUV-Konvertierung."""
        fb = FrameBuffer(2, 2, PixelFormat.YUV422)
        
        # Erzeuge ein Testmuster
        test_pattern = np.zeros((2, 2, 3), dtype=np.uint8)
        test_pattern[0, 0] = [255, 0, 0]    # Rot
        test_pattern[0, 1] = [0, 255, 0]    # Grün
        test_pattern[1, 0] = [0, 0, 255]    # Blau
        test_pattern[1, 1] = [255, 255, 0]  # Gelb
        
        # Schreibe das Testmuster
        fb.begin_frame_write()
        fb.write_pixel_data(test_pattern)
        fb.end_frame_write()
        
        # Lese das Testmuster zurück
        result = fb.get_frame_as_numpy()
        
        # Da YUV422 U und V zwischen benachbarten Pixeln teilt, gibt es Farbverluste
        # Wir überprüfen grob, ob die Farben stimmen
        
        # Rot sollte hauptsächlich Rot-Komponente haben
        self.assertTrue(result[0, 0, 0] > 200)
        self.assertTrue(result[0, 0, 1] < 100)
        self.assertTrue(result[0, 0, 2] < 100)
        
        # Grün sollte hauptsächlich Grün-Komponente haben
        self.assertTrue(result[0, 1, 0] < 100)
        self.assertTrue(result[0, 1, 1] > 200)
        self.assertTrue(result[0, 1, 2] < 100)
        
        # Blau sollte hauptsächlich Blau-Komponente haben
        self.assertTrue(result[1, 0, 0] < 100)
        self.assertTrue(result[1, 0, 1] < 100)
        self.assertTrue(result[1, 0, 2] > 200)
    
    def test_buffer_overflow_protection(self):
        """Testet den Überlaufschutz des Puffers."""
        fb = FrameBuffer(4, 4, PixelFormat.RGB888)
        
        # Versuche, über die Puffergrenzen hinaus zu schreiben
        fb.begin_frame_write()
        data = bytearray(fb.total_size_bytes + 100)  # Mehr Daten als der Puffer aufnehmen kann
        bytes_written = fb.write_pixel_data(data)
        fb.end_frame_write()
        
        # Sollte auf die Puffergröße begrenzt sein
        self.assertEqual(bytes_written, fb.total_size_bytes)
    
    def test_statistics(self):
        """Testet die Statistikfunktionen."""
        fb = FrameBuffer(320, 240, PixelFormat.RGB888)
        
        # Führe einige Operationen durch
        fb.begin_frame_write()
        fb.write_pixel_data(bytearray(100))
        fb.end_frame_write()
        
        fb.get_frame_as_numpy()
        
        # Überprüfe die Statistiken
        stats = fb.get_statistics()
        self.assertEqual(stats['width'], 320)
        self.assertEqual(stats['height'], 240)
        self.assertEqual(stats['pixel_format'], 'RGB888')
        self.assertEqual(stats['frames_processed'], 1)
        self.assertEqual(stats['frames_dropped'], 0)
        self.assertEqual(stats['write_operations'], 1)
        self.assertEqual(stats['read_operations'], 1)
    
    def test_memory_layout(self):
        """Testet das Speicherlayout mit Padding."""
        # Test mit Breite, die Padding erfordert
        fb = FrameBuffer(321, 240, PixelFormat.RGB888)
        layout = fb.get_memory_layout()
        
        raw_row_bytes = 321 * 3
        expected_aligned_row_bytes = (raw_row_bytes + 3) & ~3
        expected_padding = expected_aligned_row_bytes - raw_row_bytes
        
        self.assertEqual(layout['raw_row_bytes'], raw_row_bytes)
        self.assertEqual(layout['aligned_row_bytes'], expected_aligned_row_bytes)
        self.assertEqual(layout['padding_bytes_per_row'], expected_padding)
        self.assertEqual(layout['total_padding_bytes'], expected_padding * 240)
    
    def test_concurrent_write_protection(self):
        """Testet den Schutz vor gleichzeitigen Schreibvorgängen."""
        fb = FrameBuffer(10, 10, PixelFormat.RGB888)
        
        # Erster Schreibvorgang
        self.assertTrue(fb.begin_frame_write())
        
        # Versuch, einen weiteren Schreibvorgang zu starten, während der erste noch läuft
        self.assertFalse(fb.begin_frame_write())
        
        # Nach Beenden des ersten Schreibvorgangs sollte ein neuer möglich sein
        fb.end_frame_write()
        self.assertTrue(fb.begin_frame_write())

if __name__ == '__main__':
    unittest.main()
