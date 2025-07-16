# Hopfield Networks: Traditional Discrete & Modern Continuous

Two associative memory implementations: classical discrete binary states (1982) and modern continuous states with attention mechanism (2020).

## Shared Foundation
**Associative Memory**: Energy-based pattern storage and retrieval
```python
network.store(patterns)                                 # Store via learning rule
state, energies = network.update(noisy_state, steps)    # Retrieve via energy minimization
overlaps = network.get_overlap(state)                   # Measure pattern similarity
```

---

## Traditional Hopfield Network
**Theory**: Binary states with quadratic energy function
```
Energy: E = -1/2 s^T W s                # Quadratic form
Storage: W = (1/N) Σ_μ x^μ (x^μ)^T      # Hebbian learning  
Update: s_i = sign(Σ_j w_ij s_j)        # Asynchronous binary updates
Capacity: ~0.138 * Neurons              # Theoretical limit
```

**Implementation**:
```python
# Hebbian Storage: W = (1/N) Σ_μ x^μ (x^μ)^T
for p in patterns:
    self.weights += torch.outer(p, p)
self.weights /= self.num_neurons

# Binary Update: s_i = sign(Σ_j w_ij s_j)
local_field = torch.einsum('j,j->', self.weights[i], state)
state[i] = torch.sign(local_field)
```

---

## Modern Hopfield Network  
**Theory**: Continuous states with exponential capacity, equivalent to transformer attention
```
Energy: E = -lse(β, X^T Z) + 1/2||Z||^2 + constants
Update: Z = X softmax(β X^T Z)          # Attention mechanism
Storage: x^μ = x^μ / ||x^μ||_2           # Unit sphere normalization
Capacity: O(2^(αd))                     # Exponential in dimension
```

**Implementation**:
```python
# Attention update: Z = X softmax(β X^T Z) 
similarities = torch.matmul(self.patterns, state)
attention_weights = torch.softmax(self.beta * similarities, dim=0)  # softmax(β X^T Z)
state = torch.matmul(self.patterns.t(), attention_weights)
```

## Performance Comparison

| Aspect | Traditional         | Modern              |
|--------|---------------------|---------------------|
| **States** | Binary: {-1, +1}    | Continuous: ℝ^d     |
| **Capacity** | ~0.138N             | O(2^(αd))           |
| **Convergence** | Multiple Iterations | CCCP: One Update    |
| **Connection** | Quadratic Energy    | Attention Mechanism |


## Results
- **Capacity**: Modern networks exceed 0.138N traditional limit
- **Robustness**: Exponentially small retrieval errors with noisy inputs  
- **Applications**: Modern Hopfield layers enable pooling, memory, and attention in deep learning

## Citation

If you use my code in your research, please cite me:

```bibtex
@software{pjm2025Hopfield,
  author = {Paul J Mello},
  title = {Hopfield Networks},
  url = {https://github.com/pauljmello/Hopfield-Networks},
  year = {2025},
}
```

## References
[1] Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities  
[2] Ramsauer, H., et al. (2020). Hopfield Networks is All You Need. arXiv:2008.02217