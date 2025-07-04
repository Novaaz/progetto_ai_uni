# Makefile per progetto AI - CityLearn

PYTHON = python
SCRIPTS_DIR = scripts

.PHONY: help generate analyze

# Target di default
help:
    @echo "Makefile per progetto AI - CityLearn"
    @echo "====================================="
    @echo ""
    @echo "Comandi disponibili:"
    @echo "  make generate    - Genera nuovi modelli"
    @echo "  make analyze     - Analizza modelli esistenti"
    @echo "  make help        - Mostra questo aiuto"
    @echo ""

# Genera modelli
generate:
    @echo "🚀 Avvio generazione modelli..."
    $(PYTHON) -m $(SCRIPTS_DIR).generate_models

# Analizza modelli
analyze:
    @echo "📊 Avvio analisi modelli..."
    $(PYTHON) -m $(SCRIPTS_DIR).analyze_models