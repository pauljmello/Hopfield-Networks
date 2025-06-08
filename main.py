import os
import torch
import numpy as np

from utils import ensure_dir, get_test_patterns, create_network, test_retrieval, plot_comparison_retrieval, test_capacity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(14)


def main():
    # Configuration
    seed = 1001
    torch.manual_seed(seed)
    np.random.seed(seed)
    save_dir = 'images'
    plot_results = True
    network_types = ['traditional', 'modern']

    # Pattern Shape
    img_size = 20
    pattern_step_size = 7
    ascii_symbols = ['?', '@', 'B']

    # Hopfield Network Parameters
    max_patterns = 256
    num_neurons = img_size * img_size
    theoretical = int(0.138 * num_neurons)
    steps = 100

    # Experiment Parameters
    noise_level = 0.1
    capacity_noise = 0.1
    capacity_trials = 5

    # Modern Hopfield Network Parameter
    beta = 1.0  # inverse temperature

    # Create directories
    if plot_results:
        ensure_dir(save_dir)
        ensure_dir(f"{save_dir}/traditional")
        ensure_dir(f"{save_dir}/modern")
        ensure_dir(f"{save_dir}/comparison")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test patterns
    patterns, shape, names = get_test_patterns(ascii_symbols, img_size, device=device)
    print(f"Loaded {len(patterns)} patterns with {num_neurons} neurons each")

    # Traditional Hopfield Network
    trad_network = create_network('traditional', num_neurons, None, device)
    trad_network.store(patterns)
    trad_results = test_retrieval(trad_network, patterns, shape, names, steps, noise_level, save_dir=f"{save_dir}/traditional" if plot_results else None, prefix='traditional_')
    trad_success_rate, trad_pattern_results, trad_energies, trad_noisy = trad_results

    # Modern Hopfield Network
    modern_network = create_network('modern', num_neurons, beta, device)
    modern_network.store(patterns)
    modern_results = test_retrieval(modern_network, patterns, shape, names, steps, noise_level, save_dir=f"{save_dir}/modern" if plot_results else None, prefix='modern_')
    modern_success_rate, modern_pattern_results, modern_energies, modern_noisy = modern_results

    # Plot comparisons
    if plot_results:
        for i in range(len(patterns)):
            name = f" ({names[i]})" if names and i < len(names) else ""
            plot_comparison_retrieval(trad_noisy[i], trad_pattern_results[i], modern_pattern_results[i], trad_energies[i], modern_energies[i], shape,
                                      title=f"Pattern {i}{name} - Retrieval Comparison", save_path=f"{save_dir}/comparison/retrieval_comparison_{i}.png")

    # Test storage capacities
    _, _, _ = test_capacity(theoretical, max_patterns, steps, pattern_step_size, capacity_noise, capacity_trials, network_types, beta,
                            save_dir=f"{save_dir}/comparison" if plot_results else None, device=device)

    print("\n=== Summary ===")
    print(f"  - Storage capacity: ~0.138N = ~{theoretical} patterns")
    print(f"  - Energy function: E = -(1/2) s^T W s")
    print(f"  - Update rule: s_i = sign(Σ_j w_ij s_j)")
    print(f"  - Note:\n"
          f"        - Discrete Representations.\n")
    print("\nModern Hopfield Network:")
    print(f"  - Storage capacity: Exponential O(2^(αd)) where α < 1/(2ln2)")
    print(f"  - Energy function: E = -lse(β, X^T Z) + (1/2)||Z||² + constants")
    print(f"  - Update rule: Z = X softmax(β X^T Z)")
    print(f"  - Note:\n"
          f"        - Essentially adding an Attention Mechanism.\n"
          f"        - Continuous Representations.\n"
          f"        - Exponential storage capacity.\n")

if __name__ == "__main__":
    main()