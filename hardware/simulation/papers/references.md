# Paper References für PG-ES Sim-to-Real Optimization

## Core Papers

### 1. OpenAI Evolution Strategies
- **Titel:** Evolution Strategies as a Scalable Alternative to Reinforcement Learning
- **Autoren:** Salimans et al. (2017)
- **Link:** [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)
- **DOI:** 10.48550/arXiv.1703.03864
- **Kernidee:** Black-box optimization durch ES, parallelisierbar, gradient-free
- **Relevanz:** Grundlage für ES-Komponente unseres Hybrid-Systems

### 2. Deep Neuroevolution meets Policy Gradients
- **Titel:** Deep Neuroevolution meets Policy Gradients
- **Autoren:** Conti et al. (2018)
- **Link:** [arXiv:1712.06567](https://arxiv.org/abs/1712.06567)
- **DOI:** 10.48550/arXiv.1712.06567
- **Kernidee:** Kombination von ES und PG für bessere Sample-Effizienz
- **Relevanz:** Direkte Vorlage für unsere Hybrid-Methode

### 3. Hybrid Evolutionary Policy Gradient Methods
- **Titel:** Deep Neuroevolution for Continuous Control
- **Autoren:** Such et al. (2020)
- **Link:** [arXiv:1711.03824](https://arxiv.org/abs/1711.03824)
- **DOI:** 10.48550/arXiv.1711.03824
- **Kernidee:** Praktische Anwendung von PG+ES in kontinuierlicher Kontrolle
- **Relevanz:** Implementierungsdetails für Hardware-Parameteroptimierung

### 4. Domain Randomization for Sim-to-Real Transfer
- **Titel:** Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
- **Autoren:** Tobin et al. (2017)
- **Link:** [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- **DOI:** 10.48550/arXiv.1703.06907
- **Kernidee:** Robustheit durch Randomisierung der Simulationsparameter
- **Relevanz:** Methode für Sim-to-Real ohne echte Hardware-Daten

### 5. Deep RL That Matters
- **Titel:** Deep Reinforcement Learning that Matters
- **Autoren:** Henderson et al. (2018)
- **Link:** [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- **DOI:** 10.48550/arXiv.1709.06560
- **Kernidee:** Kritische Analyse von RL-Variabilität und Reproduzierbarkeit
- **Relevanz:** Guidelines für robuste Experimente und Logging

## Zusätzliche Referenzen

### Parameter Noise
- **Titel:** Parameter Space Noise for Exploration
- **Autoren:** Plappert et al. (2017)
- **Link:** [arXiv:1706.01905](https://arxiv.org/abs/1706.01905)
- **Relevanz:** Alternative zu ES-Noise, potentiell kombinierbar

### Code-Referenzen
- **OpenAI Baselines:** https://github.com/openai/baselines
- **ES Implementation:** https://github.com/openai/evolution-strategies-starter

## Dokumentationsstatus
- [x] Alle Haupt-Papers gesammelt
- [x] DOIs und Links verifiziert  
- [x] Kernideen zusammengefasst
- [x] Relevanz für Projekt bewertet
- [ ] PDFs heruntergeladen und archiviert
- [ ] Detaillierte Notizen zu Implementierungsdetails







































