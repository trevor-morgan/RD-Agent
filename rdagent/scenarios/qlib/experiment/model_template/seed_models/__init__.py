"""Seed models for RD-Agent evolution.

These models can be used as starting points for LLM-driven model evolution.

Available Models:
    - simple_gru: Simple 2-layer bidirectional GRU with attention pooling
    - symplectic_net: Physics-informed model with Hamiltonian dynamics
    - market_state_net: Market state detection with topological features

Usage:
    rdagent fin_model \
        --seed-model ./seed_models/symplectic_net.py \
        --seed-hypothesis "Description of model architecture" \
        --data-region us_data \
        --loop-n 5
"""
