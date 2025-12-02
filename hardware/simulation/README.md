# PG-ES Sim-to-Real Hardware Optimization

This project implements a hybrid Policy Gradient - Evolution Strategy optimization system for hardware parameter tuning with simulation-to-real transfer.

## Project Structure

```
simulation/
├── config/              # Configuration files
│   ├── config.py       # Main configuration classes
│   └── simulation_parameters.md  # Parameter documentation
├── src/                 # Source code
│   ├── hardware_simulator.py    # Physics simulation engine
│   ├── pg_optimizer.py          # Policy Gradient implementation
│   ├── es_optimizer.py          # Evolution Strategy implementation
│   └── hybrid_trainer.py       # Combined PG-ES trainer
├── papers/              # Research references
│   └── references.md    # Paper bibliography
├── logs/               # Training logs and results
└── requirements.txt    # Python dependencies
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Basic Simulation Test**
   ```bash
   python demo.py
   ```

3. **Start Training (Debug Mode)**
   ```bash
   python main.py --config debug --iterations 1000
   ```

4. **Start Production Training**
   ```bash
   python main.py --config production --iterations 10000
   ```

## Current Status

### ✅ Completed (Phase A & B)
- [x] Physical simulation parameters defined
- [x] Safety constraints implemented  
- [x] PG/ES parameter separation
- [x] Domain randomization setup
- [x] Core simulation engine
- [x] Configuration management
- [x] Paper references collected
- [x] **Policy Gradient implementation (Actor-Critic)**
- [x] **Evolution Strategy implementation (OpenAI-style)**
- [x] **Hybrid training loop with adaptive mixing**
- [x] **Complete Fail-Safe Supervisor System (7 checks)**
- [x] **Comprehensive logging and monitoring**

### 📋 Next Steps (Phase C)
- [ ] Advanced visualization dashboard
- [ ] Automated report generation
- [ ] Version control integration
- [ ] Performance benchmarking suite

## Hardware Context

This simulation is based on the PizzaBoard-RP2040 PCB design:
- Microcontroller: RP2040
- Operating voltage: 3.3V
- Temperature range: -40°C to +85°C
- Focus: Thermal management, power efficiency, longevity

## Key Features

- **Hybrid Optimization**: Combines gradient-based (PG) and gradient-free (ES) methods
- **Physics Simulation**: Realistic thermal, electrical, and mechanical modeling
- **Domain Randomization**: Sim-to-real transfer without real hardware data
- **Safety Constraints**: Hard limits prevent dangerous operating conditions
- **Comprehensive Logging**: Full traceability of optimization process

## Research Foundation

Based on key papers:
- OpenAI ES (Salimans et al., 2017)
- Deep Neuroevolution meets Policy Gradients (Conti et al., 2018)
- Domain Randomization for Sim2Real (Tobin et al., 2017)

See `papers/references.md` for complete bibliography.

## License

[Your license here]
