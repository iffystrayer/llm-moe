# 🚀 Getting Started with LLM-MoE

Welcome! This guide will get you up and running with the LLM-MoE project in just a few minutes.

## 📋 Quick Overview

This project compares **Standard Transformers** vs **Mixture of Experts (MoE)** transformers:

- **Standard**: Uses all parameters for every token (100% utilization)
- **MoE**: Uses only selected experts per token (~47% utilization, better performance!)

## ⚡ Quick Start (2 minutes)

### 1. Install Dependencies
```bash
pip install datasets transformers torchtune torchao tqdm numpy
```

### 2. Test Installation (Optional)
```bash
python offline_test.py
```
This verifies everything works without requiring internet access.

### 3. Run Full Experiment
```bash
python llm.py
```

**Expected Results:**
- Regular Transformer: Val Loss 0.1365, Accuracy 97.66%, Perplexity 1.15
- MoE Transformer: Val Loss 0.0758, Accuracy 98.57%, Perplexity 1.08
- Training time: ~7-10 minutes total

## 📊 What You'll See

```
🧪 TESTING: Regular Transformer
============================================================
Epoch 1/1: 100%|████████| 625/625 [01:50<00:00,  5.67it/s]
🏆 Final Results:
   Validation Loss: 0.1365
   Validation Accuracy: 0.9766
   Validation Perplexity: 1.15

🧪 TESTING: Mixture of Experts  
============================================================
Epoch 1/1: 100%|████████| 625/625 [03:36<00:00,  2.89it/s]
🏆 Final Results:
   Validation Loss: 0.0758
   Validation Accuracy: 0.9857
   Validation Perplexity: 1.08
```

## 🎯 Key Findings

| Model | Parameters | Active Params | Val Loss | Val Acc | Perplexity |
|-------|------------|---------------|----------|---------|------------|
| Regular | ~29M | 29M (100%) | 0.1365 | 97.66% | 1.15 |
| MoE | ~54M | ~25M (47%) | 0.0758 | 98.57% | 1.08 |

**🚀 MoE achieves better performance with fewer active parameters!**

## 🛠️ Requirements

- **Python**: 3.8+ (tested with 3.10+)
- **Memory**: 4GB+ RAM recommended
- **GPU**: Optional (will use CPU if no GPU available)
- **Internet**: Required for downloading SmolLM dataset

## 🔧 Configuration

The models are configured in `llm.py`:

```python
# Standard Transformer
ModelConfig(
    d_model=384,    # Hidden dimension
    n_heads=8,      # Attention heads
    n_layers=6,     # Transformer layers
    d_ff=1536,      # Feed-forward dimension
    max_steps=3000  # Training steps
)

# MoE Transformer  
MoEModelConfig(
    # ... same base config ...
    num_experts=8,     # Number of experts
    expert_top_k=2,    # Experts per token
    moe_layers="alternate"  # Which layers use MoE
)
```

## 📚 Next Steps

1. **Read the full tutorial**: See `TUTORIAL.md` for comprehensive guide
2. **Experiment**: Try different configurations in `llm.py`
3. **Monitor**: Use `gpu_monitor.py` for performance tracking
4. **Understand**: Explore the 13 core classes in the codebase

## 🆘 Troubleshooting

### Common Issues

**Out of Memory**:
```python
# Reduce batch size
config.batch_size = 8  # Instead of 24
```

**Slow Training**:
```python
# Reduce model size for testing
config.d_model = 256
config.max_steps = 1000
```

**Import Errors**:
```bash
pip install torch datasets transformers torchtune
```

**Network Issues**:
```bash
# Test offline first
python offline_test.py
```

## 🏆 What Makes This Special

1. **Side-by-side comparison** of architectures
2. **Real performance metrics** on same dataset
3. **Advanced techniques**: RoPE, RMSNorm, Muon optimizer
4. **Complete implementation** ready for research/learning
5. **Comprehensive documentation** for all skill levels

## 🤝 Need Help?

- Read `TUTORIAL.md` for detailed explanations
- Check `offline_test.py` for functionality verification  
- Review `llm.py` for implementation details
- Use `setup.py` for automated environment setup

**Happy experimenting! 🚀**