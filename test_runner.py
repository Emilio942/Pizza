#!/usr/bin/env python3
"""
Kompletter Test Runner für das Pizza-Projekt
Überprüft alle wichtigen Komponenten und Skripte
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import importlib.util
from typing import List, Dict, Tuple
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProjectTester:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            'pytest_results': {},
            'script_tests': {},
            'import_tests': {},
            'integration_tests': {},
            'summary': {}
        }
        
    def check_environment(self) -> bool:
        """Überprüft die Python-Umgebung und Dependencies"""
        logger.info("🔍 Überprüfe Python-Umgebung...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            logger.info(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                logger.info("✅ Virtual Environment aktiv")
            else:
                logger.warning("⚠️ Kein Virtual Environment erkannt")
            
            # Check requirements
            requirements_file = self.project_root / "config" / "requirements.txt"
            if requirements_file.exists():
                logger.info("✅ Requirements-Datei gefunden")
                return True
            else:
                logger.error("❌ Requirements-Datei nicht gefunden")
                return False
                
        except Exception as e:
            logger.error(f"❌ Umgebungscheck fehlgeschlagen: {e}")
            return False
    
    def run_pytest_tests(self) -> Dict:
        """Führt alle pytest-Tests aus"""
        logger.info("🧪 Starte pytest-Tests...")
        
        results = {}
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest", 
                "tests/", 
                "-v", 
                "--tb=short",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:reports/coverage"
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True, 
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            results['exit_code'] = result.returncode
            results['stdout'] = result.stdout
            results['stderr'] = result.stderr
            results['success'] = result.returncode == 0
            
            if result.returncode == 0:
                logger.info("✅ Alle pytest-Tests erfolgreich")
            else:
                logger.error(f"❌ pytest-Tests fehlgeschlagen (Exit Code: {result.returncode})")
                logger.error(f"Fehler: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ pytest-Tests timeout (>5 Minuten)")
            results['success'] = False
            results['error'] = "Timeout"
        except Exception as e:
            logger.error(f"❌ pytest-Ausführung fehlgeschlagen: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_key_imports(self) -> Dict:
        """Testet wichtige Importe"""
        logger.info("📦 Teste wichtige Importe...")
        
        key_modules = [
            'torch',
            'torchvision', 
            'opencv-python',
            'numpy',
            'PIL',
            'scipy',
            'sklearn',
            'matplotlib',
            'pandas',
            'stable_baselines3',
            'diffusers',
            'transformers'
        ]
        
        results = {}
        
        for module in key_modules:
            try:
                # Handle special cases
                if module == 'opencv-python':
                    import cv2
                    results[module] = {'success': True, 'version': cv2.__version__}
                elif module == 'PIL':
                    from PIL import Image
                    results[module] = {'success': True, 'version': 'imported'}
                else:
                    imported_module = __import__(module)
                    version = getattr(imported_module, '__version__', 'unknown')
                    results[module] = {'success': True, 'version': version}
                    
                logger.info(f"✅ {module} - Version: {results[module]['version']}")
                
            except ImportError as e:
                logger.error(f"❌ {module} - Import fehlgeschlagen: {e}")
                results[module] = {'success': False, 'error': str(e)}
            except Exception as e:
                logger.error(f"❌ {module} - Unerwarteter Fehler: {e}")
                results[module] = {'success': False, 'error': str(e)}
                
        return results
    
    def test_critical_scripts(self) -> Dict:
        """Testet kritische Skripte"""
        logger.info("🔧 Teste kritische Skripte...")
        
        critical_scripts = [
            'scripts/classify_images.py',
            'scripts/spatial_model_validation.py',
            'scripts/end_to_end_testing.py',
            'scripts/benchmark_preprocessing_optimization_fixed.py',
            'scripts/test_spatial_pizza_classification.py'
        ]
        
        results = {}
        
        for script_path in critical_scripts:
            full_path = self.project_root / script_path
            script_name = Path(script_path).name
            
            logger.info(f"Testing {script_name}...")
            
            try:
                if not full_path.exists():
                    results[script_name] = {
                        'success': False, 
                        'error': 'Datei nicht gefunden'
                    }
                    logger.error(f"❌ {script_name} - Datei nicht gefunden")
                    continue
                
                # Test syntax by attempting to compile
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                compile(content, full_path, 'exec')
                
                results[script_name] = {
                    'success': True, 
                    'message': 'Syntax OK'
                }
                logger.info(f"✅ {script_name} - Syntax OK")
                
            except SyntaxError as e:
                results[script_name] = {
                    'success': False, 
                    'error': f'Syntax Error: {e}'
                }
                logger.error(f"❌ {script_name} - Syntax Error: {e}")
            except Exception as e:
                results[script_name] = {
                    'success': False, 
                    'error': str(e)
                }
                logger.error(f"❌ {script_name} - Fehler: {e}")
                
        return results
    
    def test_data_availability(self) -> Dict:
        """Überprüft verfügbare Daten"""
        logger.info("📊 Überprüfe Datenverfügbarkeit...")
        
        data_paths = [
            'data/',
            'models/',
            'augmented_pizza/',
            'test_data/',
            'config/'
        ]
        
        results = {}
        
        for data_path in data_paths:
            full_path = self.project_root / data_path
            
            if full_path.exists():
                if full_path.is_dir():
                    file_count = len(list(full_path.rglob('*')))
                    results[data_path] = {
                        'success': True,
                        'type': 'directory',
                        'file_count': file_count
                    }
                    logger.info(f"✅ {data_path} - {file_count} Dateien")
                else:
                    results[data_path] = {
                        'success': True,
                        'type': 'file'
                    }
                    logger.info(f"✅ {data_path} - Datei vorhanden")
            else:
                results[data_path] = {
                    'success': False,
                    'error': 'Pfad nicht gefunden'
                }
                logger.warning(f"⚠️ {data_path} - Nicht gefunden")
                
        return results
    
    def run_integration_tests(self) -> Dict:
        """Führt Integrationstests aus"""
        logger.info("🔗 Starte Integrationstests...")
        
        integration_scripts = [
            'scripts/spatial_integration_tests.py',
            'scripts/end_to_end_testing.py'
        ]
        
        results = {}
        
        for script_path in integration_scripts:
            full_path = self.project_root / script_path
            script_name = Path(script_path).name
            
            if not full_path.exists():
                results[script_name] = {
                    'success': False,
                    'error': 'Skript nicht gefunden'
                }
                continue
                
            try:
                cmd = [sys.executable, str(full_path)]
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minutes timeout
                )
                
                results[script_name] = {
                    'success': result.returncode == 0,
                    'exit_code': result.returncode,
                    'stdout': result.stdout[-500:],  # Last 500 chars
                    'stderr': result.stderr[-500:] if result.stderr else None
                }
                
                if result.returncode == 0:
                    logger.info(f"✅ {script_name} - Erfolgreich")
                else:
                    logger.error(f"❌ {script_name} - Fehlgeschlagen")
                    
            except subprocess.TimeoutExpired:
                results[script_name] = {
                    'success': False,
                    'error': 'Timeout'
                }
                logger.error(f"❌ {script_name} - Timeout")
            except Exception as e:
                results[script_name] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"❌ {script_name} - Fehler: {e}")
                
        return results
    
    def generate_summary(self) -> Dict:
        """Generiert eine Zusammenfassung der Testergebnisse"""
        logger.info("📋 Generiere Zusammenfassung...")
        
        summary = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'success_rate': 0.0,
            'categories': {}
        }
        
        # Count results from all categories
        for category, tests in self.results.items():
            if category == 'summary':
                continue
                
            category_success = 0
            category_total = 0
            
            if category == 'pytest_results':
                category_total = 1
                category_success = 1 if tests.get('success', False) else 0
            elif isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    category_total += 1
                    if isinstance(test_result, dict) and test_result.get('success', False):
                        category_success += 1
            
            summary['categories'][category] = {
                'total': category_total,
                'successful': category_success,
                'success_rate': (category_success / category_total * 100) if category_total > 0 else 0
            }
            
            summary['total_tests'] += category_total
            summary['successful_tests'] += category_success
        
        summary['failed_tests'] = summary['total_tests'] - summary['successful_tests']
        summary['success_rate'] = (summary['successful_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        
        return summary
    
    def run_all_tests(self):
        """Führt alle Tests aus"""
        logger.info("🚀 Starte komplette Testübersicht...")
        
        start_time = time.time()
        
        # 1. Environment check
        if not self.check_environment():
            logger.error("❌ Umgebungscheck fehlgeschlagen - Stoppe Tests")
            return
            
        # 2. Import tests
        self.results['import_tests'] = self.test_key_imports()
        
        # 3. Data availability
        self.results['data_tests'] = self.test_data_availability()
        
        # 4. Script syntax tests
        self.results['script_tests'] = self.test_critical_scripts()
        
        # 5. pytest tests
        self.results['pytest_results'] = self.run_pytest_tests()
        
        # 6. Integration tests
        self.results['integration_tests'] = self.run_integration_tests()
        
        # 7. Generate summary
        self.results['summary'] = self.generate_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"⏱️ Tests abgeschlossen in {duration:.2f} Sekunden")
        
        self.print_final_report()
    
    def print_final_report(self):
        """Druckt den finalen Testbericht"""
        print("\n" + "="*80)
        print("🍕 PIZZA-PROJEKT TEST ÜBERSICHT")
        print("="*80)
        
        summary = self.results['summary']
        
        print(f"\n📊 GESAMTERGEBNIS:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Erfolgreich: {summary['successful_tests']}")
        print(f"   Fehlgeschlagen: {summary['failed_tests']}")
        print(f"   Erfolgsrate: {summary['success_rate']:.1f}%")
        
        # Category breakdown
        print(f"\n📋 KATEGORIE ÜBERSICHT:")
        for category, stats in summary['categories'].items():
            status = "✅" if stats['success_rate'] >= 80 else "⚠️" if stats['success_rate'] >= 50 else "❌"
            print(f"   {status} {category:20}: {stats['successful']:2}/{stats['total']:2} ({stats['success_rate']:5.1f}%)")
        
        # Detailed results
        print(f"\n🔍 DETAILLIERTE ERGEBNISSE:")
        
        # Import tests
        print(f"\n   📦 Import Tests:")
        for module, result in self.results['import_tests'].items():
            status = "✅" if result['success'] else "❌"
            print(f"      {status} {module:20}: {result.get('version', result.get('error', 'N/A'))}")
        
        # Script tests
        print(f"\n   🔧 Script Tests:")
        for script, result in self.results['script_tests'].items():
            status = "✅" if result['success'] else "❌"
            print(f"      {status} {script:30}: {result.get('message', result.get('error', 'N/A'))}")
        
        # Pytest results
        print(f"\n   🧪 Pytest Ergebnisse:")
        pytest_result = self.results['pytest_results']
        if pytest_result:
            status = "✅" if pytest_result.get('success') else "❌"
            print(f"      {status} pytest: Exit Code {pytest_result.get('exit_code', 'N/A')}")
        
        # Integration tests
        print(f"\n   🔗 Integration Tests:")
        for test, result in self.results['integration_tests'].items():
            status = "✅" if result['success'] else "❌"
            print(f"      {status} {test:30}: {result.get('error', 'OK')}")
        
        # Overall status
        print(f"\n" + "="*80)
        if summary['success_rate'] >= 90:
            print("🎉 AUSGEZEICHNET! Alle Systeme funktionieren einwandfrei!")
        elif summary['success_rate'] >= 75:
            print("👍 GUT! Die meisten Systeme funktionieren, kleine Verbesserungen möglich.")
        elif summary['success_rate'] >= 50:
            print("⚠️ MITTELMÄSSIG! Einige wichtige Probleme müssen behoben werden.")
        else:
            print("❌ KRITISCH! Viele Systeme funktionieren nicht - dringende Reparaturen nötig!")
        print("="*80)

if __name__ == "__main__":
    project_root = "/home/emilio/Documents/ai/pizza"
    tester = ProjectTester(project_root)
    tester.run_all_tests()
