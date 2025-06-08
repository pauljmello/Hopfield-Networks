import torch


class ModernHopfieldNetwork:
    """Modern Hopfield Network with continuous states as described in
    Introduced in 2020: "Hopfield Networks is All You Need" - https://arxiv.org/abs/2008.02217.

    Key features:
    - Continuous state vectors
    - Exponential storage capacity
    - Retrieval update
    - Attention mechanism
    
    - Energy function: E = -lse(β, X^T Z) + (1/2)Z^T Z + β^-1 ln N + (1/2)M^2
    - Update rule: Z_new = X softmax(βX^T Z)
    """

    def __init__(self, num_neurons, beta, device='cpu'):
        self.num_neurons, self.beta, self.device = num_neurons, beta, device
        self.patterns = None
        self.num_patterns = 0
        self.max_norm = 0

    def store(self, patterns):
        patterns = patterns.to(self.device).float()

        # Pattern normalization
        patterns = patterns / torch.norm(patterns, dim=1, keepdim=True)   # Unit sphere normalization: x^μ = x^μ / ||x^μ||_2

        self.patterns = patterns
        self.num_patterns = patterns.shape[0]
        self.max_norm = torch.max(torch.norm(patterns, dim=1))
        return self  # Chaining

    def energy(self, state):
        """E = -lse(β, X^T Z) + (1/2)Z^T Z + β^(-1) ln N + (1/2)M²"""
        state = state.to(self.device)
        similarities = torch.matmul(self.patterns, state)
        lse_term = (1.0 / self.beta) * torch.logsumexp(self.beta * similarities, dim=0)
        constant_term = (1.0 / self.beta) * torch.log(torch.tensor(self.num_patterns, device=self.device).float()) + 0.5 * (self.max_norm ** 2)
        return -lse_term + 0.5 * torch.sum(state * state) + constant_term

    def update(self, state, steps):
        """Update state using X softmax(βX^T Z)"""
        state = state.to(self.device).clone()
        prev_state = state.clone()
        similarities = torch.matmul(self.patterns, state)
        energies = [self.energy(state).item()]

        for _ in range(steps):
            attention_weights = torch.softmax(self.beta * similarities, dim=0)
            state = torch.matmul(self.patterns.t(), attention_weights)  # Update State
            if torch.allclose(state, prev_state, atol=1e-6):
                return state, energies
            energies.append(self.energy(state).item())
            prev_state = state.clone()

        return state, energies

    def get_overlap(self, state):
        return torch.tensor([torch.sum(state * p) / self.num_neurons for p in self.patterns])