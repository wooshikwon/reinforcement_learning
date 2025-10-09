"""
Script to plot DAgger vs BC learning curves
"""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def extract_scalar_data(event_file, scalar_name='Eval_AverageReturn'):
    """Extract scalar data from TensorBoard event file"""
    ea = EventAccumulator(event_file)
    ea.Reload()

    if scalar_name not in ea.Tags()['scalars']:
        print(f"Warning: {scalar_name} not found in {event_file}")
        return [], []

    events = ea.Scalars(scalar_name)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_dagger_comparison():
    """Plot DAgger vs BC comparison for both environments"""

    # Define experiment directories
    experiments = {
        'Ant-v4': {
            'BC': 'data/q1_bc_ant_Ant-v4_08-10-2025_21-08-37',
            'DAgger': 'data/q2_dagger_ant_Ant-v4_08-10-2025_21-50-34',
            'expert': 4713.65  # From the experiment output
        },
        'HalfCheetah-v4': {
            'BC': 'data/q1_bc_halfcheetah_HalfCheetah-v4_08-10-2025_21-09-29',
            'DAgger': 'data/q2_dagger_halfcheetah_HalfCheetah-v4_08-10-2025_21-53-29',
            'expert': 4205.78  # From the experiment output
        }
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (env_name, exp_data) in enumerate(experiments.items()):
        ax = axes[idx]

        # Extract BC data
        bc_event_file = None
        for file in os.listdir(exp_data['BC']):
            if file.startswith('events.out.tfevents'):
                bc_event_file = os.path.join(exp_data['BC'], file)
                break

        if bc_event_file:
            bc_steps, bc_returns = extract_scalar_data(bc_event_file)
            ax.plot(bc_steps, bc_returns, 'o-', label='Behavioral Cloning',
                   color='#1f77b4', linewidth=2, markersize=6)

        # Extract DAgger data
        dagger_event_file = None
        for file in os.listdir(exp_data['DAgger']):
            if file.startswith('events.out.tfevents'):
                dagger_event_file = os.path.join(exp_data['DAgger'], file)
                break

        if dagger_event_file:
            dagger_steps, dagger_returns = extract_scalar_data(dagger_event_file)
            ax.plot(dagger_steps, dagger_returns, 's-', label='DAgger',
                   color='#ff7f0e', linewidth=2, markersize=6)

        # Plot expert performance line
        expert_perf = exp_data['expert']
        if bc_event_file and dagger_event_file:
            max_step = max(max(bc_steps), max(dagger_steps))
            ax.axhline(y=expert_perf, color='green', linestyle='--',
                      linewidth=2, label=f'Expert ({expert_perf:.2f})')

        # Formatting
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Evaluation Return', fontsize=12)
        ax.set_title(f'{env_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Set y-axis limits for better visualization
        if bc_event_file and dagger_event_file:
            all_returns = bc_returns + dagger_returns
            y_min = min(all_returns) * 0.95
            y_max = max(max(all_returns), expert_perf) * 1.05
            ax.set_ylim([y_min, y_max])

    plt.suptitle('DAgger vs Behavioral Cloning: Learning Curves',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    output_file = 'dagger_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")

    # Also save as PDF for LaTeX
    output_pdf = 'dagger_comparison.pdf'
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {output_pdf}")

    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for env_name, exp_data in experiments.items():
        print(f"\n{env_name}:")
        print("-" * 40)

        # BC stats
        bc_event_file = None
        for file in os.listdir(exp_data['BC']):
            if file.startswith('events.out.tfevents'):
                bc_event_file = os.path.join(exp_data['BC'], file)
                break

        if bc_event_file:
            _, bc_returns = extract_scalar_data(bc_event_file)
            if bc_returns:
                bc_final = bc_returns[-1] if len(bc_returns) > 0 else 0
                bc_mean = np.mean(bc_returns)
                bc_std = np.std(bc_returns)
                print(f"BC Final Return: {bc_final:.2f}")
                print(f"BC Mean Return: {bc_mean:.2f} ± {bc_std:.2f}")

        # DAgger stats
        dagger_event_file = None
        for file in os.listdir(exp_data['DAgger']):
            if file.startswith('events.out.tfevents'):
                dagger_event_file = os.path.join(exp_data['DAgger'], file)
                break

        if dagger_event_file:
            _, dagger_returns = extract_scalar_data(dagger_event_file)
            if dagger_returns:
                dagger_final = dagger_returns[-1] if len(dagger_returns) > 0 else 0
                dagger_mean = np.mean(dagger_returns)
                dagger_std = np.std(dagger_returns)
                print(f"DAgger Final Return: {dagger_final:.2f}")
                print(f"DAgger Mean Return: {dagger_mean:.2f} ± {dagger_std:.2f}")

        expert_perf = exp_data['expert']
        print(f"Expert Performance: {expert_perf:.2f}")

        if bc_event_file and dagger_event_file and bc_returns and dagger_returns:
            bc_final = bc_returns[-1]
            dagger_final = dagger_returns[-1]
            bc_pct = (bc_final / expert_perf) * 100
            dagger_pct = (dagger_final / expert_perf) * 100
            improvement = dagger_final - bc_final
            improvement_pct = ((dagger_final - bc_final) / bc_final) * 100

            print(f"\nBC vs Expert: {bc_pct:.1f}%")
            print(f"DAgger vs Expert: {dagger_pct:.1f}%")
            print(f"DAgger Improvement over BC: +{improvement:.2f} ({improvement_pct:+.1f}%)")

if __name__ == "__main__":
    plot_dagger_comparison()
