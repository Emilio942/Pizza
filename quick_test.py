#!/usr/bin/env python3
"""
Schneller Test Runner für das Pizza-Projekt
Überprüft alle wichtigen Komponenten
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_imports():
    """Teste wichtige Importe"""
    print("📦 Teste wichtige Importe...")
    
    modules = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('stable_baselines3', 'stable-baselines3'),
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers')
    ]
    
    results = {}
    
    for import_name, package_name in modules:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'imported')
            results[package_name] = {'success': True, 'version': version}
            print(f"   ✅ {package_name:20}: {version}")
        except ImportError:
            results[package_name] = {'success': False}
            print(f"   ❌ {package_name:20}: Import fehlgeschlagen")
        except Exception as e:
            results[package_name] = {'success': False}
            print(f"   ❌ {package_name:20}: {str(e)[:50]}")
    
    return results

def check_scripts():
    """Teste kritische Skripte"""
    print("🔧 Teste kritische Skripte...")
    
    scripts = [
        'scripts/classify_images.py',
        'scripts/spatial_model_validation.py', 
        'scripts/end_to_end_testing.py',
        'scripts/test_spatial_pizza_classification.py',
        'scripts/benchmark_preprocessing_optimization_fixed.py'
    ]
    
    results = {}
    project_root = Path('/home/emilio/Documents/ai/pizza')
    
    for script_path in scripts:
        script_name = Path(script_path).name
        full_path = project_root / script_path
        
        try:
            if not full_path.exists():
                results[script_name] = {'success': False, 'error': 'Nicht gefunden'}
                print(f"   ❌ {script_name:30}: Datei nicht gefunden")
                continue
                
            # Test syntax
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, full_path, 'exec')
            results[script_name] = {'success': True}
            print(f"   ✅ {script_name:30}: Syntax OK")
            
        except SyntaxError as e:
            results[script_name] = {'success': False, 'error': f'Syntax Error: {e}'}
            print(f"   ❌ {script_name:30}: Syntax Error")
        except Exception as e:
            results[script_name] = {'success': False, 'error': str(e)}
            print(f"   ❌ {script_name:30}: {str(e)[:30]}")
    
    return results

def check_data():
    """Überprüfe Datenverfügbarkeit"""
    print("📊 Überprüfe Datenverfügbarkeit...")
    
    project_root = Path('/home/emilio/Documents/ai/pizza')
    paths = ['data/', 'models/', 'augmented_pizza/', 'test_data/', 'config/']
    
    results = {}
    
    for path in paths:
        full_path = project_root / path
        
        if full_path.exists() and full_path.is_dir():
            file_count = len(list(full_path.rglob('*')))
            results[path] = {'success': True, 'file_count': file_count}
            print(f"   ✅ {path:20}: {file_count} Dateien")
        else:
            results[path] = {'success': False}
            print(f"   ❌ {path:20}: Nicht gefunden")
    
    return results

def run_quick_pytest():
    """Führe schnelle pytest Tests aus"""
    print("🧪 Starte schnelle pytest Tests...")
    
    try:
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "-x",  # Stop on first failure
            "--maxfail=3"  # Stop after 3 failures
        ]
        
        result = subprocess.run(
            cmd, 
            cwd='/home/emilio/Documents/ai/pizza',
            capture_output=True, 
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        success = result.returncode == 0
        
        if success:
            print("   ✅ pytest Tests: Erfolgreich")
        else:
            print(f"   ❌ pytest Tests: Fehlgeschlagen (Code: {result.returncode})")
            if result.stderr:
                print(f"      Fehler: {result.stderr[:200]}...")
                
        return {'success': success, 'exit_code': result.returncode}
        
    except subprocess.TimeoutExpired:
        print("   ⚠️ pytest Tests: Timeout")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"   ❌ pytest Tests: {str(e)}")
        return {'success': False, 'error': str(e)}

def main():
    """Hauptfunktion"""
    print("🍕 PIZZA-PROJEKT SCHNELL-TEST")
    print("="*50)
    
    start_time = time.time()
    
    # Tests durchführen
    import_results = check_imports()
    script_results = check_scripts()
    data_results = check_data()
    pytest_results = run_quick_pytest()
    
    # Statistiken berechnen
    def count_success(results):
        if not results:
            return 0, 0
        total = len(results)
        successful = sum(1 for r in results.values() if r.get('success', False))
        return successful, total
    
    import_success, import_total = count_success(import_results)
    script_success, script_total = count_success(script_results)
    data_success, data_total = count_success(data_results)
    pytest_success = 1 if pytest_results.get('success', False) else 0
    pytest_total = 1
    
    total_success = import_success + script_success + data_success + pytest_success
    total_tests = import_total + script_total + data_total + pytest_total
    
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Ergebnisse anzeigen
    print("\n" + "="*50)
    print("📊 TESTERGEBNISSE")
    print("="*50)
    
    print(f"⏱️  Testdauer: {duration:.1f} Sekunden")
    print(f"📈 Gesamtergebnis: {total_success}/{total_tests} ({success_rate:.1f}%)")
    
    print(f"\n📦 Importe: {import_success}/{import_total} ({import_success/import_total*100:.1f}%)")
    print(f"🔧 Skripte: {script_success}/{script_total} ({script_success/script_total*100:.1f}%)")
    print(f"📊 Daten: {data_success}/{data_total} ({data_success/data_total*100:.1f}%)")
    print(f"🧪 pytest: {pytest_success}/{pytest_total} ({pytest_success/pytest_total*100:.1f}%)")
    
    # Gesamtbewertung
    print("\n" + "="*50)
    if success_rate >= 90:
        print("🎉 AUSGEZEICHNET! Alles funktioniert super!")
    elif success_rate >= 75:
        print("👍 GUT! Die meisten Systeme funktionieren.")
    elif success_rate >= 50:
        print("⚠️ MITTELMÄSSIG! Einige Probleme vorhanden.")
    else:
        print("❌ KRITISCH! Viele Probleme - Reparaturen nötig!")
    print("="*50)
    
    # Detaillierte Probleme
    if success_rate < 100:
        print("\n🔍 PROBLEME:")
        
        for name, result in import_results.items():
            if not result.get('success', False):
                print(f"   ❌ Import {name}: {result.get('error', 'Fehlgeschlagen')}")
        
        for name, result in script_results.items():
            if not result.get('success', False):
                print(f"   ❌ Script {name}: {result.get('error', 'Fehlgeschlagen')}")
        
        for name, result in data_results.items():
            if not result.get('success', False):
                print(f"   ❌ Daten {name}: Nicht verfügbar")
        
        if not pytest_results.get('success', False):
            print(f"   ❌ pytest: {pytest_results.get('error', 'Fehlgeschlagen')}")

if __name__ == "__main__":
    main()
