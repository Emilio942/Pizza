# CI/CD-Pipeline für Pizza-Erkennungssystem

Diese Pipeline automatisiert den gesamten Prozess vom Modelltraining bis zur Firmware-Erstellung für das Pizza-Erkennungssystem.

## Übersicht

Die CI/CD-Pipeline führt folgende Schritte automatisch aus:

1. **Vorbereitung**: Prüfung der Datensatzintegrität und Grundlegende Tests
2. **Modelltraining**: Training des Pizza-Erkennungsmodells mit optimierten Hyperparametern
3. **Modellquantisierung**: Konvertierung des Float32-Modells zu einem Int8-Modell für den RP2040
4. **C-Code-Generierung**: Umwandlung des Modells in C-Code für die Embedded-Plattform
5. **Firmware-Erstellung**: Kompilierung der Firmware mit dem RP2040 SDK
6. **Tests**: Umfassende Tests des Modells und der generierten Artefakte
7. **Benachrichtigung**: E-Mail und Slack-Benachrichtigungen bei erfolgreichem Abschluss

## Auslöser der Pipeline

Die Pipeline wird automatisch ausgeführt bei:

- Push auf den `main` oder `master` Branch (nur bei Änderungen in relevanten Verzeichnissen)
- Pull Requests auf den `main` oder `master` Branch
- Manuellem Auslösen über die GitHub-Oberfläche

## Artefakte

Nach einem erfolgreichen Durchlauf werden folgende Artefakte generiert:

- **trained-model**: Das trainierte PyTorch-Modell (Float32)
- **quantized-model**: Das quantisierte Modell (Int8)
- **model-c-code**: Der generierte C-Code für den RP2040
- **rp2040-firmware**: Die kompilierte UF2-Firmware-Datei

Die Artefakte werden für 7 Tage in GitHub gespeichert und können über die GitHub Actions UI heruntergeladen werden.

## Qualitätsschranken

Die Pipeline definiert folgende Qualitätsschwellen:

- Mindestgenauigkeit des Modells: 70%
- Maximale Modellgröße (Int8): 200 KB

Wenn eine dieser Schwellen nicht erreicht wird, schlägt die Pipeline fehl und sendet keine Benachrichtigungen.

## Benachrichtigungen

Bei erfolgreicher Ausführung sendet die Pipeline Benachrichtigungen an:

- Die konfigurierte E-Mail-Adresse (`team@pizza-detection.example.com`)
- Den konfigurierten Slack-Kanal (falls eingerichtet)

## Konfiguration

### Umgebungsvariablen

Die Pipeline verwendet folgende Umgebungsvariablen:

- `PYTHON_VERSION`: Python-Version für die Ausführung (Standard: 3.10)
- `MODEL_OUTPUT_DIR`: Ausgabeverzeichnis für Modelle (Standard: models)
- `FIRMWARE_OUTPUT_DIR`: Ausgabeverzeichnis für Firmware (Standard: models/rp2040_export)
- `NOTIFICATION_EMAIL`: E-Mail-Adresse für Benachrichtigungen

### Secrets

Die folgenden Secrets müssen in den GitHub Repository-Einstellungen konfiguriert werden:

- `MAIL_USERNAME`: Benutzername für den SMTP-Server
- `MAIL_PASSWORD`: Passwort für den SMTP-Server
- `SLACK_WEBHOOK_URL`: Webhook-URL für Slack-Benachrichtigungen (optional)

## Manuelle Ausführung

Um die Pipeline manuell auszuführen:

1. Gehe zu "Actions" in deinem GitHub Repository
2. Wähle "Pizza Detection Model CI/CD Pipeline" aus der Liste
3. Klicke auf "Run workflow"
4. Wähle den Branch aus
5. Klicke auf "Run workflow"

## Fehlerbehebung

Wenn die Pipeline fehlschlägt, überprüfe:

1. Die Logs der fehlgeschlagenen Jobs
2. Die Datensatzintegrität
3. Die Konfiguration der Secrets
4. Die verfügbaren Ressourcen in den GitHub Actions Runnern

## Weiterentwicklung

Mögliche zukünftige Verbesserungen der Pipeline:

- Integration von A/B-Tests für verschiedene Modellarchitekturen
- Automatische Modelloptimierung über Neural Architecture Search
- Deployment der Firmware auf Test-Hardware
- Erweiterte Testabdeckung einschließlich Hardware-in-the-Loop-Tests