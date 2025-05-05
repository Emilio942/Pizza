# Makefile für CLAHE-Implementierung (Pizza-Erkennungssystem)
# 
# Dieses Makefile kompiliert die CLAHE-Implementierung und führt Tests durch,
# um sicherzustellen, dass sie korrekt funktioniert und die Ressourcenanforderungen
# des RP2040-Mikrocontrollers erfüllt.

CC = gcc
RP2040_CC = arm-none-eabi-gcc
CFLAGS = -Wall -Wextra -g -O2
RP2040_CFLAGS = -mcpu=cortex-m0plus -mthumb -O2 -DNDEBUG

# Verzeichnisse
SRC_DIR = .
TEST_DIR = ../../tests/clahe
BUILD_DIR = ../../output/clahe_resources

# Dateien
SOURCES = $(SRC_DIR)/pizza_clahe.c
TEST_SOURCES = $(TEST_DIR)/test_clahe.c
EXECUTABLE = $(BUILD_DIR)/clahe_test
RP2040_OBJECTS = $(BUILD_DIR)/pizza_clahe_rp2040.o

# Erstelle Verzeichnisse
$(shell mkdir -p $(BUILD_DIR))

# Standard-Ziel
all: test rp2040

# Nur Kompilieren (ohne Tests)
compile: $(SOURCES)
	$(CC) $(CFLAGS) -c $(SOURCES) -o $(BUILD_DIR)/pizza_clahe.o

# RP2040-Variante kompilieren
rp2040: $(SOURCES)
	@echo "Kompiliere für RP2040-Mikrocontroller..."
	@if command -v $(RP2040_CC) > /dev/null 2>&1; then \
		$(RP2040_CC) $(RP2040_CFLAGS) -c $(SOURCES) -o $(RP2040_OBJECTS); \
		echo "RAM & Flash Nutzung:"; \
		arm-none-eabi-size $(RP2040_OBJECTS); \
	else \
		echo "WARNUNG: RP2040-Compiler nicht gefunden. Überspringen..."; \
	fi

# Tests kompilieren und ausführen
test: $(SOURCES) $(TEST_SOURCES)
	@echo "Kompiliere und führe Tests aus..."
	@if [ -d $(TEST_DIR) ]; then \
		$(CC) $(CFLAGS) $(SOURCES) $(TEST_SOURCES) -o $(EXECUTABLE); \
		$(EXECUTABLE); \
	else \
		echo "WARNUNG: Testverzeichnis nicht gefunden. Überspringe Tests..."; \
	fi

# Ressourcenanforderungen analysieren
resources: compile
	@echo "Analysiere Ressourcenanforderungen..."
	@if [ -f ../../scripts/simulate_clahe_resources.py ]; then \
		python3 ../../scripts/simulate_clahe_resources.py --output $(BUILD_DIR); \
	else \
		echo "WARNUNG: Ressourcensimulations-Skript nicht gefunden."; \
	fi

# Aufräumen
clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(EXECUTABLE)

.PHONY: all compile test rp2040 resources clean