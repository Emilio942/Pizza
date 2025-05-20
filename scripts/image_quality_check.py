#!/usr/bin/env python3
"""
Image Quality Control Script for Pizza Recognition System

This script helps with manual quality control of augmented and generated images:
1. Selects random samples from augmented and synthetic image directories
2. Generates an HTML report with thumbnail views for manual inspection
3. Allows marking images for deletion
4. Processes marked images and removes them from the dataset

Usage:
    python image_quality_check.py --samples 50 --output output/quality_control
"""

import os
import sys
import argparse
import random
import json
import shutil
from pathlib import Path
from datetime import datetime
import webbrowser
import glob
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library not found. Install it with: pip install pillow")
    sys.exit(1)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Quality Control for Pizza Dataset')
    parser.add_argument('--image-dirs', nargs='+', default=['augmented_pizza'],
                        help='Directories containing augmented/generated images relative to project root')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of random samples to select per category')
    parser.add_argument('--output', default='output/quality_control',
                        help='Output directory for the HTML report and control files')
    parser.add_argument('--check-existing', action='store_true',
                        help='Check existing control data and show only unprocessed images')
    parser.add_argument('--process-marked', action='store_true',
                        help='Process marked images from a previous run (delete marked files)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (do not actually delete files)')
    return parser.parse_args()

def collect_image_paths(image_dirs, exclude_paths=None):
    """Collect all image paths from the specified directories."""
    if exclude_paths is None:
        exclude_paths = set()
    
    image_paths = defaultdict(list)
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for base_dir in image_dirs:
        base_path = os.path.join(project_root, base_dir)
        if not os.path.exists(base_path):
            continue
            
        # Iterate through class directories
        for class_dir in os.listdir(base_path):
            class_path = os.path.join(base_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Skip backup directories
            if 'backup' in class_dir.lower():
                continue
                
            # Find all images in this class directory
            for ext in image_exts:
                files = glob.glob(os.path.join(class_path, f'*{ext}'))
                for file_path in files:
                    if file_path not in exclude_paths:
                        # Use class_dir as the category name
                        image_paths[class_dir].append(file_path)
    
    return image_paths

def select_random_samples(image_paths, num_samples):
    """Select random samples from each category."""
    selected_samples = {}
    
    for class_name, paths in image_paths.items():
        # If there are fewer images than requested samples, take all of them
        if len(paths) <= num_samples:
            selected_samples[class_name] = paths
        else:
            selected_samples[class_name] = random.sample(paths, num_samples)
    
    return selected_samples

def generate_thumbnails(selected_samples, output_dir):
    """Generate thumbnails of selected images."""
    thumbnail_dir = os.path.join(output_dir, 'thumbnails')
    os.makedirs(thumbnail_dir, exist_ok=True)
    
    thumbnail_map = {}
    for class_name, paths in selected_samples.items():
        class_thumbnail_dir = os.path.join(thumbnail_dir, class_name)
        os.makedirs(class_thumbnail_dir, exist_ok=True)
        
        for i, img_path in enumerate(paths):
            try:
                # Generate a thumbnail name based on the original filename
                filename = os.path.basename(img_path)
                thumbnail_path = os.path.join(class_thumbnail_dir, filename)
                
                # Create and save the thumbnail
                with Image.open(img_path) as img:
                    img_copy = img.copy()
                    img_copy.thumbnail((200, 200))
                    img_copy.save(thumbnail_path)
                
                thumbnail_map[img_path] = thumbnail_path
            except Exception as e:
                print(f"Error creating thumbnail for {img_path}: {e}")
    
    return thumbnail_map

def generate_html_report(selected_samples, thumbnail_map, output_dir, run_id):
    """Generate an HTML report with image thumbnails and controls for marking images."""
    html_path = os.path.join(output_dir, 'quality_control_report.html')
    
    # Prepare paths relative to the HTML file
    relative_thumbnail_map = {}
    for orig_path, thumb_path in thumbnail_map.items():
        relative_thumbnail_map[orig_path] = os.path.relpath(thumb_path, output_dir)
    
    # Create a dictionary to pass as JavaScript data
    js_data = {
        'runId': run_id,
        'images': [],
        'outputDir': output_dir
    }
    
    for class_name, paths in selected_samples.items():
        for img_path in paths:
            if img_path in thumbnail_map:
                js_data['images'].append({
                    'id': os.path.basename(img_path),
                    'path': img_path,
                    'thumbnailPath': relative_thumbnail_map[img_path],
                    'className': class_name
                })
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    js_data_str = json.dumps(js_data)
    
    # Create HTML content with JavaScript
    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pizza Dataset - Bildqualitätskontrolle</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .class-section {{
            margin-bottom: 40px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
            position: relative;
        }}
        .image-card.marked {{
            border: 3px solid #e74c3c;
            opacity: 0.7;
        }}
        .image-card:hover {{
            transform: scale(1.03);
        }}
        .thumbnail {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background-color: #f8f9fa;
        }}
        .info {{
            padding: 10px;
            background-color: #f8f9fa;
        }}
        .info h4 {{
            margin: 0 0 5px 0;
            font-size: 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .controls {{
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            position: sticky;
            bottom: 0;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }}
        .btn {{
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            margin-right: 10px;
            margin-bottom: 5px;
        }}
        .btn-primary {{
            background-color: #3498db;
            color: white;
        }}
        .btn-danger {{
            background-color: #e74c3c;
            color: white;
        }}
        .btn-success {{
            background-color: #2ecc71;
            color: white;
        }}
        .badge {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            background-color: #e74c3c;
            color: white;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
        }}
        .stats {{
            margin-top: 10px;
            font-size: 14px;
        }}
        .checkbox-container {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .checkbox-container input {{
            margin-right: 10px;
        }}
        .filter-controls {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .clickable {{
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pizza Dataset - Bildqualitätskontrolle</h1>
        <p>Generiert am: {timestamp}</p>
        <p>Diese Seite zeigt zufällig ausgewählte Bilder zur Qualitätskontrolle. Klicke auf ein Bild, um es für die Löschung zu markieren oder die Markierung aufzuheben.</p>
        
        <div class="filter-controls">
            <h3>Filter</h3>
            <div class="checkbox-container">
                <input type="checkbox" id="filter-marked" onchange="filterImages()">
                <label for="filter-marked">Nur markierte Bilder anzeigen</label>
            </div>
            
            <h3>Klassen</h3>
            <div id="class-filters">
                <!-- Wird dynamisch befüllt -->
            </div>
        </div>
        
        <div id="content">
            <!-- Wird dynamisch befüllt -->
        </div>
        
        <div class="controls">
            <div class="stats">
                <p><strong>Markierte Bilder:</strong> <span id="marked-count">0</span> von <span id="total-count">0</span></p>
            </div>
            <button class="btn btn-danger" onclick="markAllVisible()">Alle sichtbaren Bilder markieren</button>
            <button class="btn btn-primary" onclick="unmarkAllVisible()">Markierungen aufheben (sichtbar)</button>
            <button class="btn btn-success" onclick="saveMarkedImages()">Markierte Bilder speichern</button>
        </div>
    </div>
    
    <script>
        // Bilddaten aus Python
        const imageData = {js_data_str};
        
        // Status der markierten Bilder
        let markedImages = new Set();
        let visibleImages = new Set();
        
        // Klassen-Filter
        let activeClassFilters = new Set();
        
        // Initialisiere die Seite
        function initPage() {{
            // Gesamtzahl der Bilder anzeigen
            document.getElementById('total-count').textContent = imageData.images.length;
            
            // Sammle alle einzigartigen Klassen
            const classes = [...new Set(imageData.images.map(img => img.className))];
            
            // Aktiviere alle Klassenfilter initial
            classes.forEach(className => activeClassFilters.add(className));
            
            // Erstelle Klassenfilter
            const classFiltersContainer = document.getElementById('class-filters');
            classes.forEach(className => {{
                const classCount = imageData.images.filter(img => img.className === className).length;
                
                const checkbox = document.createElement('div');
                checkbox.className = 'checkbox-container';
                checkbox.innerHTML = `
                    <input type="checkbox" id="filter-class-${{className}}" checked onchange="toggleClassFilter('${{className}}')">
                    <label for="filter-class-${{className}}">${{className}} (${{classCount}})</label>
                `;
                classFiltersContainer.appendChild(checkbox);
            }});
            
            // Erstelle den Inhalt
            renderContent();
        }}
        
        // Rendere den gesamten Inhalt
        function renderContent() {{
            const contentContainer = document.getElementById('content');
            contentContainer.innerHTML = '';
            
            // Gruppiere Bilder nach Klasse
            const imagesByClass = {{}};
            imageData.images.forEach(img => {{
                if (!imagesByClass[img.className]) {{
                    imagesByClass[img.className] = [];
                }}
                imagesByClass[img.className].push(img);
            }});
            
            // Erstelle Abschnitte für jede Klasse
            Object.entries(imagesByClass).forEach(([className, images]) => {{
                // Überspringe Klasse, wenn sie nicht im Filter ist
                if (!activeClassFilters.has(className)) {{
                    return;
                }}
                
                const classSection = document.createElement('div');
                classSection.className = 'class-section';
                classSection.id = `class-${{className}}`;
                
                const classTitle = document.createElement('h2');
                classTitle.textContent = `Klasse: ${{className}}`;
                classSection.appendChild(classTitle);
                
                const imageGrid = document.createElement('div');
                imageGrid.className = 'image-grid';
                
                // Erstelle eine Karte für jedes Bild
                images.forEach(img => {{
                    const imgCard = createImageCard(img);
                    
                    // Wenn nur markierte Bilder angezeigt werden sollen und dieses Bild nicht markiert ist, verstecke es
                    const filterMarked = document.getElementById('filter-marked').checked;
                    if (filterMarked && !markedImages.has(img.path)) {{
                        imgCard.style.display = 'none';
                    }} else {{
                        visibleImages.add(img.path);
                    }}
                    
                    imageGrid.appendChild(imgCard);
                }});
                
                classSection.appendChild(imageGrid);
                contentContainer.appendChild(classSection);
            }});
            
            // Aktualisiere die Zähler
            updateCounters();
        }}
        
        // Erstelle eine Bildkarte
        function createImageCard(img) {{
            const card = document.createElement('div');
            card.className = 'image-card clickable';
            card.dataset.path = img.path;
            
            if (markedImages.has(img.path)) {{
                card.classList.add('marked');
                
                const badge = document.createElement('div');
                badge.className = 'badge';
                badge.textContent = 'Löschen';
                card.appendChild(badge);
            }}
            
            const thumbnail = document.createElement('img');
            thumbnail.className = 'thumbnail';
            thumbnail.src = img.thumbnailPath;
            thumbnail.alt = img.id;
            thumbnail.title = 'Klicken, um als zu löschend zu markieren';
            card.appendChild(thumbnail);
            
            const info = document.createElement('div');
            info.className = 'info';
            
            const title = document.createElement('h4');
            title.textContent = img.id;
            title.title = img.path; // Vollständiger Pfad als Tooltip
            info.appendChild(title);
            
            const className = document.createElement('p');
            className.textContent = `Klasse: ${{img.className}}`;
            info.appendChild(className);
            
            card.appendChild(info);
            
            // Event-Listener zum Markieren/Entmarkieren des Bildes
            card.addEventListener('click', () => toggleMark(img.path, card));
            
            return card;
        }}
        
        // Markiere oder entmarkiere ein Bild
        function toggleMark(imagePath, card) {{
            if (markedImages.has(imagePath)) {{
                markedImages.delete(imagePath);
                card.classList.remove('marked');
                
                // Entferne das Badge
                const badge = card.querySelector('.badge');
                if (badge) {{
                    card.removeChild(badge);
                }}
            }} else {{
                markedImages.add(imagePath);
                card.classList.add('marked');
                
                // Füge Badge hinzu
                const badge = document.createElement('div');
                badge.className = 'badge';
                badge.textContent = 'Löschen';
                card.appendChild(badge);
            }}
            
            // Aktualisiere den Zähler
            updateCounters();
        }}
        
        // Markiere alle sichtbaren Bilder
        function markAllVisible() {{
            document.querySelectorAll('.image-card').forEach(card => {{
                const imagePath = card.dataset.path;
                if (card.style.display !== 'none' && !markedImages.has(imagePath)) {{
                    markedImages.add(imagePath);
                    card.classList.add('marked');
                    
                    // Füge Badge hinzu, falls nicht vorhanden
                    if (!card.querySelector('.badge')) {{
                        const badge = document.createElement('div');
                        badge.className = 'badge';
                        badge.textContent = 'Löschen';
                        card.appendChild(badge);
                    }}
                }}
            }});
            
            // Aktualisiere den Zähler
            updateCounters();
        }}
        
        // Entmarkiere alle sichtbaren Bilder
        function unmarkAllVisible() {{
            document.querySelectorAll('.image-card').forEach(card => {{
                const imagePath = card.dataset.path;
                if (card.style.display !== 'none' && markedImages.has(imagePath)) {{
                    markedImages.delete(imagePath);
                    card.classList.remove('marked');
                    
                    // Entferne das Badge
                    const badge = card.querySelector('.badge');
                    if (badge) {{
                        card.removeChild(badge);
                    }}
                }}
            }});
            
            // Aktualisiere den Zähler
            updateCounters();
        }}
        
        // Filtere Bilder basierend auf dem "Nur markierte Bilder anzeigen" Checkbox
        function filterImages() {{
            visibleImages.clear();
            const filterMarked = document.getElementById('filter-marked').checked;
            
            document.querySelectorAll('.image-card').forEach(card => {{
                const imagePath = card.dataset.path;
                const img = imageData.images.find(img => img.path === imagePath);
                
                // Prüfe, ob die Klasse des Bildes im aktiven Filter ist
                const isClassFiltered = activeClassFilters.has(img.className);
                
                if (filterMarked) {{
                    // Wenn "Nur markierte" aktiviert ist und das Bild nicht markiert ist, verstecke es
                    if (!markedImages.has(imagePath) || !isClassFiltered) {{
                        card.style.display = 'none';
                    }} else {{
                        card.style.display = 'block';
                        visibleImages.add(imagePath);
                    }}
                }} else {{
                    // Wenn "Nur markierte" nicht aktiviert ist, zeige es an, wenn die Klasse im Filter ist
                    if (isClassFiltered) {{
                        card.style.display = 'block';
                        visibleImages.add(imagePath);
                    }} else {{
                        card.style.display = 'none';
                    }}
                }}
            }});
            
            // Aktualisiere Klassenüberschriften
            Object.keys(imageData.images.reduce((acc, img) => {{
                acc[img.className] = true;
                return acc;
            }}, {{}})).forEach(className => {{
                const classSection = document.getElementById(`class-${{className}}`);
                if (classSection) {{
                    // Zähle sichtbare Bilder in dieser Klasse
                    const visibleInClass = Array.from(document.querySelectorAll(`.image-card[data-path^="{project_root}/augmented_pizza/${{className}}/"]`))
                        .filter(card => card.style.display !== 'none').length;
                    
                    if (visibleInClass === 0) {{
                        classSection.style.display = 'none';
                    }} else {{
                        classSection.style.display = 'block';
                    }}
                }}
            }});
        }}
        
        // Aktiviere/Deaktiviere einen Klassenfilter
        function toggleClassFilter(className) {{
            const checkbox = document.getElementById(`filter-class-${{className}}`);
            
            if (checkbox.checked) {{
                activeClassFilters.add(className);
            }} else {{
                activeClassFilters.delete(className);
            }}
            
            // Rendere den Inhalt neu
            renderContent();
            
            // Wende den "Nur markierte" Filter an, falls aktiviert
            if (document.getElementById('filter-marked').checked) {{
                filterImages();
            }}
        }}
        
        // Aktualisiere den Zähler markierter Bilder
        function updateCounters() {{
            document.getElementById('marked-count').textContent = markedImages.size;
        }}
        
        // Speichere markierte Bilder in eine JSON-Datei
        function saveMarkedImages() {{
            // Erstelle ein spezielles Markierungsobjekt
            const markedData = {{
                runId: imageData.runId,
                timestamp: new Date().toISOString(),
                markedImages: Array.from(markedImages)
            }};
            
            // Erstelle eine versteckte Form zum Senden der Daten
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = `./save_marked_images.py?run_id=${{imageData.runId}}`;
            form.style.display = 'none';
            
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'marked_data';
            input.value = JSON.stringify(markedData);
            
            form.appendChild(input);
            document.body.appendChild(form);
            
            // Sende die Form
            form.submit();
        }}
        
        // Initialisiere die Seite beim Laden
        window.onload = initPage;
    </script>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Create a simple Python script to handle form submission
    save_script_path = os.path.join(output_dir, 'save_marked_images.py')
    save_script_content = """#!/usr/bin/env python3
import os
import sys
import json
import cgi
import cgitb
import urllib.parse
from datetime import datetime

# Enable detailed error reporting
cgitb.enable()

# Get query string parameters
query_string = os.environ.get('QUERY_STRING', '')
params = urllib.parse.parse_qs(query_string)
run_id = params.get('run_id', ['unknown'])[0]

print("Content-Type: text/html")
print()

try:
    # Parse form data
    form = cgi.FieldStorage()
    
    if 'marked_data' not in form:
        raise ValueError("No marked data provided")
    
    # Get marked data from form
    marked_data = json.loads(form.getvalue('marked_data'))
    
    # Save the marked data to a file
    output_file = f"marked_images_{run_id}.json"
    
    with open(output_file, 'w') as f:
        json.dump(marked_data, f, indent=2)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Marked Images Saved</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 50px; }}
        .success {{ color: green; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Markierte Bilder gespeichert</h1>
        <p class="success">Die markierten Bilder wurden erfolgreich gespeichert!</p>
        <p>{len(marked_data['markedImages'])} Bilder wurden für die Löschung markiert.</p>
        <p>Um die markierten Bilder zu löschen, führen Sie folgenden Befehl aus:</p>
        <pre>python scripts/image_quality_check.py --process-marked --output {os.path.dirname(os.path.abspath(output_file))}</pre>
        <p><a href="javascript:history.back()">Zurück zur Qualitätskontrolle</a></p>
    </div>
</body>
</html>
'''
    print(html_content)
    
except Exception as e:
    # Show error page
    error_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 50px; }}
        .error {{ color: red; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fehler beim Speichern der markierten Bilder</h1>
        <p class="error">{str(e)}</p>
        <p><a href="javascript:history.back()">Zurück zur Qualitätskontrolle</a></p>
    </div>
</body>
</html>
'''
    print(error_html)
"""
    
    with open(save_script_path, 'w', encoding='utf-8') as f:
        f.write(save_script_content)
    
    # Make the save script executable
    os.chmod(save_script_path, 0o755)
    
    return html_path

def process_marked_images(output_dir, dry_run=False):
    """Process previously marked images by deleting them from the dataset."""
    marked_files = glob.glob(os.path.join(output_dir, "marked_images_*.json"))
    
    if not marked_files:
        print("No marked image files found.")
        return 0, []
    
    deleted_count = 0
    deleted_paths = []
    
    for marked_file in marked_files:
        try:
            with open(marked_file, 'r') as f:
                marked_data = json.load(f)
            
            # Get the list of marked images
            marked_images = marked_data.get('markedImages', [])
            
            # Delete each marked image
            for img_path in marked_images:
                if os.path.exists(img_path):
                    if not dry_run:
                        os.remove(img_path)
                    deleted_count += 1
                    deleted_paths.append(img_path)
                    action = "Would delete" if dry_run else "Deleted"
                    print(f"{action}: {img_path}")
                else:
                    print(f"Warning: Marked image not found: {img_path}")
            
            # Rename the processed file to indicate it's been handled
            if not dry_run:
                processed_file = marked_file.replace('.json', '_processed.json')
                os.rename(marked_file, processed_file)
            
        except Exception as e:
            print(f"Error processing {marked_file}: {e}")
    
    return deleted_count, deleted_paths

def main():
    """Main function."""
    args = parse_args()
    
    # Convert relative paths to absolute paths
    output_dir = os.path.join(project_root, args.output)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If processing marked images, handle that and exit
    if args.process_marked:
        deleted_count, deleted_paths = process_marked_images(output_dir, args.dry_run)
        action = "Would delete" if args.dry_run else "Successfully deleted"
        if deleted_count > 0:
            print(f"\n{action} {deleted_count} marked images.")
            
            # Generate a report of deleted images
            if not args.dry_run:
                report_path = os.path.join(output_dir, f"deletion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(report_path, 'w') as f:
                    f.write(f"Deletion Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total deleted images: {deleted_count}\n\n")
                    f.write("Deleted Images:\n")
                    for path in deleted_paths:
                        f.write(f"- {path}\n")
                
                print(f"Deletion report saved to: {report_path}")
        else:
            print("No images were deleted.")
        return
    
    # Generate a unique run ID for this quality control session
    run_id = f"qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set of previously processed image paths
    exclude_paths = set()
    
    # If checking existing control data, load previously processed images
    if args.check_existing:
        marked_files = glob.glob(os.path.join(output_dir, "marked_images_*.json"))
        for marked_file in marked_files:
            try:
                with open(marked_file, 'r') as f:
                    marked_data = json.load(f)
                exclude_paths.update(marked_data.get('markedImages', []))
            except Exception as e:
                print(f"Warning: Could not read {marked_file}: {e}")
    
    # Collect all image paths
    print("Collecting image paths...")
    image_paths = collect_image_paths(args.image_dirs, exclude_paths)
    
    # Count total images found
    total_images = sum(len(paths) for paths in image_paths.values())
    print(f"Found {total_images} images across {len(image_paths)} categories.")
    
    if total_images == 0:
        print("No images found. Exiting.")
        return
    
    # Select random samples
    print(f"Selecting up to {args.samples} random samples per category...")
    selected_samples = select_random_samples(image_paths, args.samples)
    
    # Count total selected samples
    total_selected = sum(len(paths) for paths in selected_samples.values())
    print(f"Selected {total_selected} images for quality control.")
    
    # Generate thumbnails
    print("Generating thumbnails...")
    thumbnail_map = generate_thumbnails(selected_samples, output_dir)
    
    # Generate HTML report
    print("Generating HTML report...")
    html_path = generate_html_report(selected_samples, thumbnail_map, output_dir, run_id)
    
    # Open the report in a browser
    print(f"\nQuality control report generated: {html_path}")
    print(f"Opening the report in your default browser...")
    try:
        webbrowser.open(f"file://{html_path}")
    except:
        print(f"Could not open browser automatically. Please open the report manually: {html_path}")
    
    print("\nInstructions:")
    print("1. Review the images in the browser")
    print("2. Click on images to mark them for deletion")
    print("3. Click 'Save Marked Images' when done")
    print("4. Run this script with --process-marked flag to delete the marked images:")
    print(f"   python scripts/image_quality_check.py --process-marked --output {args.output}")

if __name__ == "__main__":
    main()
