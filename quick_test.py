#!/usr/bin/env python3
"""
Quick test script to verify LLM-MoE functionality
This runs a minimal test with reduced parameters for faster execution
"""

import torch
import torch.nn.functional as F
import math
import sys
import os

# Add the current directory to path so we can import from llm.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm import (
    ModelConfig, MoEModelConfig, 
    MinimalLLM, MoEMinimalLLM,
    TextTokenDataset, Muon
)
from transformers import AutoTokenizer

def quick_test():
    """Run a quick functionality test"""
    print("🚀 Running LLM-MoE Quick Test")
    print("=" * 50)
    
    # Test tokenizer
    print("📝 Testing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_text = "Hello world! This is a test."
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenizer works: '{test_text}' -> {len(tokens)} tokens")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return False
    
    # Create minimal configs for testing
    print("\n🔧 Creating test configurations...")
    
    # Very small model for quick testing
    test_config = ModelConfig(
        d_model=64,           # Much smaller
        n_heads=4,            # Fewer heads
        n_layers=2,           # Fewer layers
        d_ff=128,             # Smaller FFN
        batch_size=2,         # Small batch
        max_steps=10,         # Very few steps
        max_seq_len=32,       # Short sequences
        vocab_size=tokenizer.vocab_size,
        eval_interval=5,
        eval_steps=2,
        use_amp=False         # Disable AMP for simplicity
    )
    
    moe_test_config = MoEModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        batch_size=2,
        max_steps=10,
        max_seq_len=32,
        vocab_size=tokenizer.vocab_size,
        eval_interval=5,
        eval_steps=2,
        use_amp=False,
        num_experts=4,         # Few experts
        expert_top_k=2,        # Simple routing
        moe_layers="last"      # Only last layer
    )
    
    # Create dummy data
    print("📊 Creating test data...")
    dummy_tokens = tokens * 20  # Repeat to get enough data
    dataset = TextTokenDataset(dummy_tokens, test_config.max_seq_len)
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Test standard model
    print("\n🧪 Testing Standard Transformer...")
    try:
        model = MinimalLLM(test_config)
        print(f"✅ Created standard model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        x = torch.randint(0, test_config.vocab_size, (2, test_config.max_seq_len))
        y = torch.randint(0, test_config.vocab_size, (2, test_config.max_seq_len))
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, test_config.vocab_size), y.view(-1))
        
        print(f"✅ Forward pass works: loss = {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("✅ Backward pass works")
        
    except Exception as e:
        print(f"❌ Standard model failed: {e}")
        return False
    
    # Test MoE model
    print("\n🧠 Testing MoE Transformer...")
    try:
        moe_model = MoEMinimalLLM(moe_test_config)
        total_params = sum(p.numel() for p in moe_model.parameters())
        print(f"✅ Created MoE model with {total_params:,} parameters")
        
        # Test forward pass with aux loss
        x = torch.randint(0, moe_test_config.vocab_size, (2, moe_test_config.max_seq_len))
        y = torch.randint(0, moe_test_config.vocab_size, (2, moe_test_config.max_seq_len))
        
        logits, aux_loss = moe_model(x, return_aux_loss=True)
        main_loss = F.cross_entropy(logits.view(-1, moe_test_config.vocab_size), y.view(-1))
        
        print(f"✅ MoE forward pass works: main_loss = {main_loss.item():.4f}")
        if aux_loss is not None:
            print(f"   Auxiliary loss: {aux_loss.item():.4f}")
        
        # Test backward pass
        total_loss = main_loss + (aux_loss * 0.01 if aux_loss is not None else 0)
        total_loss.backward()
        print("✅ MoE backward pass works")
        
    except Exception as e:
        print(f"❌ MoE model failed: {e}")
        return False
    
    # Test optimizer
    print("\n⚡ Testing Muon optimizer...")
    try:
        optimizer = Muon(model.parameters(), lr=0.01)
        
        # Simple optimization step
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, test_config.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        print("✅ Muon optimizer works")
        
    except Exception as e:
        print(f"❌ Muon optimizer failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("\n📋 Summary:")
    print(f"   • Standard model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   • MoE model: {sum(p.numel() for p in moe_model.parameters()):,} parameters")
    print(f"   • Tokenizer vocabulary: {tokenizer.vocab_size:,} tokens")
    print(f"   • Test dataset: {len(dataset)} samples")
    
    print("\n✅ Ready to run full experiments!")
    print("Next step: python llm.py")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)