import torch


class HopfieldNetwork:
    """Traditional Hopfield Network: E = -1/2 s^T W s, s_i = sign(Σ w_ij s_j)

    Mathematical Foundation:
    - Energy: E = -1/2 Σ_ij w_ij s_i s_j  # quadratic form
    - Update: s_i = sign(h_i) where h_i = Σ_j w_ij s_j  # local field
    - Storage: W = (1/N) Σ_μ x^μ (x^μ)^T  # Hebbian rule
    - Capacity: ~0.138N patterns (Hopfield, 1982)
    """

    def __init__(self, num_neurons, device='cpu'):
        self.num_neurons = num_neurons
        self.weights = torch.zeros((num_neurons, num_neurons), device=device)
        self.device = device
        self.patterns = None

    def store(self, patterns):
        """Hebbian rule: W = (1/P) ∑_μ Z^μ (Z^μ)^T"""
        patterns = patterns.to(self.device)

        self.patterns = patterns.clone()
        self.weights.zero_()

        # Hebbian learning rule
        for p in patterns:
            self.weights += torch.outer(p, p)

        self.weights /= self.num_neurons
        self.weights.fill_diagonal_(0)  # Remove self-connections

        return self

    def energy(self, state):
        """Energy function: E = -1/2 s^T W s"""
        return -0.5 * torch.sum(state * (self.weights @ state))  # Quadratic form

    def update(self, state, steps):
        """Asynchronous update: s_i = sign(Σ_j w_ij s_j) for random i"""
        state = state.to(self.device).clone()
        energies = [self.energy(state).item()]

        for _ in range(steps):
            i = torch.randint(0, self.num_neurons, (1,)).item()  # random neuron selection
            local_field = torch.einsum('j,j->', self.weights[i], state)  # h_i = Σ_j w_ij s_j
            state[i] = torch.sign(local_field)  # binary activation: s_i = sign(h_i)
            energies.append(self.energy(state).item())

        return state, energies

    def get_overlap(self, state):
        """Pattern overlap: m_μ = (1/N) Σ_i s_i x^μ_i"""
        return torch.tensor([torch.sum(state * p) / self.num_neurons for p in self.patterns])