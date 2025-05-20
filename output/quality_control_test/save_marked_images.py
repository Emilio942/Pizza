#!/usr/bin/env python3
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
