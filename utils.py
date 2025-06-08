import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.gridspec import GridSpec

from hopfield import HopfieldNetwork
from modernhopfield import ModernHopfieldNetwork


def evaluate_retrieval(network, pattern, steps, noise_level):
    noisy = add_noise(pattern, noise_level)

    # Update network
    is_traditional = isinstance(network, HopfieldNetwork)

    if is_traditional:
        result, energy = network.update(noisy, steps)
    else:
        result, energy = network.update(noisy, steps)

    # Get best match
    overlaps = network.get_overlap(result)
    best_match = torch.argmax(torch.abs(overlaps)).item() if len(overlaps) > 0 else -1

    return {'noisy': noisy, 'result': result, 'energy': energy, 'best_match': best_match}


def create_network(network_type, neurons, beta, device):
    networks = {'traditional': lambda: HopfieldNetwork(neurons, device),  # Make Traditional Hopfield Network
                'modern': lambda: ModernHopfieldNetwork(neurons, beta, device=device)}  # Make Modern Hopfield Network
    network_creator = networks.get(network_type)
    return network_creator()


def test_retrieval(network, patterns, shape, names, steps, noise, save_dir, prefix):
    """Test Hopfield network pattern retrieval"""
    eval_results = [evaluate_retrieval(network, pattern, steps, noise) for pattern in patterns]

    # Compute success rate
    successes = [i for i, res in enumerate(eval_results) if res['best_match'] == i]
    successful = len(successes)
    success_rate = successful / len(patterns)

    # Results
    results = [res['result'] for res in eval_results]
    energies = [res['energy'] for res in eval_results]
    noisy_patterns = [res['noisy'] for res in eval_results]

    # Plot
    if save_dir:
        for i, res in enumerate(eval_results):
            name = f" ({names[i]})" if names and i < len(names) else ""
            title = f"Pattern {name}"
            plot_result(res['noisy'], res['result'], res['energy'], shape, title, save_path=f"{save_dir}/{prefix}retrieval_{i}.png")

    print(f"Successful retrievals: {successful}/{len(patterns)} ({100 * success_rate:.1f}%)")
    return success_rate, results, energies, noisy_patterns


def evaluate_capacity(network_type, neurons, n_patterns, steps, noise, trial_count, beta, device):
    success_count = 0
    total_tests = n_patterns * trial_count

    for _ in range(trial_count):
        patterns = 2 * (torch.rand(n_patterns, neurons, device=device) > 0.5).float() - 1  # Generate random patterns

        network = create_network(network_type, neurons, beta, device)
        network.store(patterns)

        # Test
        for i, pattern in enumerate(patterns):
            eval_result = evaluate_retrieval(network, pattern, steps, noise)
            if eval_result['best_match'] == i:
                success_count += 1

    return success_count / total_tests


def test_capacity(neurons, max_patterns, steps, step_size, noise, trial_count, network_types, beta, save_dir, device):
    theoretical = int(0.138 * neurons)
    pattern_counts = list(range(2, theoretical, step_size)) + list(range(theoretical, max_patterns + 1, step_size))

    print(f"First 5 Pattern Measurement Points: {pattern_counts[:5]} | Theoretical Capacity: {theoretical}")

    success_rates = {nt: [] for nt in network_types}

    for n_patterns in pattern_counts:
        print(f"\nTesting with {n_patterns} patterns...")

        for network_type in network_types:
            rate = evaluate_capacity(network_type, neurons, n_patterns, steps, noise, trial_count, beta, device)
            success_rates[network_type].append(rate)
            print(f"  {network_type.capitalize()}: {rate:.4f}")

    # Plot results
    if save_dir:
        if len(network_types) > 1 and 'traditional' in network_types and 'modern' in network_types:
            plot_capacity_comparison(pattern_counts, success_rates['traditional'], success_rates['modern'], neurons, save_path=f"{save_dir}/capacity_comparison.png")
        else:
            for nt in network_types:
                plot_capacity(pattern_counts, success_rates[nt], theoretical, save_path=f"{save_dir}/capacity_{nt}.png")

    return pattern_counts, success_rates, theoretical


def char_to_binary_grid(char, size):
    """Convert ASCII character to binary grid automatically"""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", size)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    img_array = np.array(img)
    binary_grid = np.where(img_array < 128, 1, 0)

    return binary_grid.tolist()


def get_test_patterns(chars, img_size, device='cpu'):
    """Generate patterns from ASCII characters"""
    patterns = []

    for char in chars:
        binary_grid = char_to_binary_grid(char, img_size)
        pattern_tensor = torch.tensor(binary_grid, dtype=torch.float32).flatten()
        patterns.append(pattern_tensor)

    patterns = torch.stack(patterns).to(device)
    patterns = 2 * patterns - 1
    shape = (img_size, img_size)

    return patterns, shape, chars


def add_noise(pattern, noise_level):
    """Flip random bits according to probability noise_level"""
    mask = torch.rand_like(pattern) < noise_level
    return pattern * (1 - 2 * mask)  # Flip bits where mask is True


def plot_result(input_pattern, output_pattern, energy, shape, title, save_path):
    """Plot retrieval results"""
    if shape is None:
        n = input_pattern.numel()
        side = int(np.sqrt(n))
        shape = (side, side if side ** 2 == n else side + 1)

    has_energy = energy is not None and len(energy) > 0
    ncols = 3 if has_energy else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Handle single subplot case
    if ncols == 1:
        axes = [axes]

    # Plot input pattern - fix tensor handling
    axes[0].imshow(input_pattern.cpu().detach().reshape(shape),  cmap='binary')
    axes[0].set_title('Input')
    axes[0].axis('off')

    # Plot output pattern - fix tensor handling
    axes[1].imshow(output_pattern.cpu().detach().reshape(shape),  cmap='binary')
    axes[1].set_title('Output')
    axes[1].axis('off')

    # Plot energy history if provided
    if has_energy:
        axes[2].plot(energy, 'b-', linewidth=2)
        axes[2].set_title('Energy')
        axes[2].set_xlabel('Update')
        axes[2].set_ylabel('Energy')
        axes[2].grid(True)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_capacity(x, y, theoretical, save_path):
    """Plot network capacity test results"""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o-', linewidth=2)

    if theoretical:
        plt.axvline(x=theoretical, color='r', linestyle='--', label=f'Theoretical capacity: {theoretical}')

    plt.axhline(y=0.5, color='g', linestyle=':', label='50% success rate')
    plt.xlabel('Number of patterns')
    plt.ylabel('Retrieval success rate')
    plt.title('Hopfield Network Storage Capacity')
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_retrieval(noisy, trad_result, modern_result, trad_energy, modern_energy, shape, title, save_path):
    """Plot retrieval comparison of traditional and modern Hopfield Networks"""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 4, figure=fig)

    # Original pattern with noise
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(noisy.cpu().reshape(shape), cmap='binary')
    ax1.set_title('Noisy Input')
    ax1.axis('off')

    # Traditional result
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(trad_result.cpu().reshape(shape), cmap='binary')
    ax2.set_title('Traditional Hopfield')
    ax2.axis('off')

    # Modern result
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(modern_result.cpu().reshape(shape), cmap='binary')
    ax3.set_title('Modern Hopfield')
    ax3.axis('off')

    # Energy comparison
    ax4 = fig.add_subplot(gs[0, 3])
    if trad_energy and modern_energy:
        ax4.plot(trad_energy, 'b-', label='Traditional')
        ax4.plot(modern_energy, 'r-', label='Modern')
        ax4.set_title('Energy Dynamics')
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Energy')
        ax4.legend()
        ax4.grid(True)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_capacity_comparison(pattern_counts, trad_rates, modern_rates, theoretical, save_path):
    """Plot capacity comparison of traditional and modern Hopfield Networks"""
    plt.figure(figsize=(10, 6))

    if trad_rates:
        plt.plot(pattern_counts, trad_rates, 'bo-', linewidth=2, label='Traditional')

    if modern_rates:
        plt.plot(pattern_counts, modern_rates, 'rs-', linewidth=2, label='Modern')

    if theoretical:
        plt.axvline(x=theoretical, color='g', linestyle='--', label=f'Theoretical Capacity: {theoretical}')

    plt.xlabel('Number of Patterns')
    plt.ylabel('Retrieval Success Rate')
    plt.title('Hopfield Network Storage Capacity Comparison')
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def ensure_dir(path):
    """Ensure directory paths"""
    os.makedirs(path, exist_ok=True)