import matplotlib.pyplot as plt
import matplotlib;matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import os,pathlib

def plot_qfi_matrices(qfi_matrix, plot_label="Some event", save_path=None,plot='average'):
    """
    Plot QFI matrix for one class of events.
    
    Args:
        qfi_matrix: Array of shape (3N, 3N) - QFI matrix where N is number of qubits
        save_path: Optional path to save the plots
    """
    if plot=='average':
        qfi_matrix = qfi_matrix.mean(axis=0)
    
    if save_path:
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Get matrix dimensions
    N_params = qfi_matrix.shape[0]
    N_qubits = N_params // 3
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Color scheme  
    colors = ['#0066FF', 'white', '#FF0066']  # Blue -> White -> Red
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
    
    # Set up normalization
    norm = matplotlib.colors.Normalize(vmin=-0.25, vmax=0.25)

    # Plot the matrix
    im = ax.matshow(qfi_matrix, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('QFI Value')
    
    # Add dark grid lines to highlight 3x3 blocks (between qubit blocks)
    for i in range(1, N_qubits):
        ax.axhline(y=3*i - 0.5, color='black', linewidth=2)
        ax.axvline(x=3*i - 0.5, color='black', linewidth=2)
    
    # Set tick positions
    rotation_positions = [i+0.5 for i in range(3*N_qubits)]
    ax.set_xticks(rotation_positions)
    ax.set_yticks(rotation_positions)
    
    # Remove tick labels (we'll add them manually between ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Turn off minor ticks and top/right axis labels
    ax.tick_params(which='minor', length=0)
    ax.tick_params(top=False, labeltop=False, right=False, labelright=False)
    
    # Make tick marks smaller
    ax.tick_params(axis='both', length=3, width=0.5, which='major')
    
    # Add rotation labels between ticks
    rotation_labels = ['$R_Z$', '$R_Y$', '$R_X$'] * N_qubits
    label_positions = [i for i in range(3*N_qubits)]  # Between ticks
    
    for pos, label in zip(label_positions, rotation_labels):
        ax.text(pos, 3*N_qubits+0.1, label, ha='center', va='top', fontsize=8, transform=ax.transData)
        ax.text(-0.7, pos, label, ha='right', va='center', fontsize=8, transform=ax.transData)
    
    # Add qubit number labels (away from axis, at center of each 3x3 block)
    qubit_positions = [1 + 3*i for i in range(N_qubits)]
    qubit_labels = [str(i) for i in range(N_qubits)]
    
    # Add qubit numbers manually as text (away from axis)
    for pos, label in zip(qubit_positions, qubit_labels):
        ax.text(pos, -0.3, label, ha='center', va='top', fontsize=12, 
                fontweight='bold', transform=ax.transData)
        ax.text(-1.4, pos, label, ha='right', va='center', fontsize=12,
                fontweight='bold', transform=ax.transData)
    
    # Set axis labels
    ax.set_xlabel('Qubit Number', labelpad=35)
    ax.set_ylabel('Qubit Number', labelpad=35)
    plt.title(plot_label, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Print statistics
    print(f"QFI Matrix Range: [{qfi_matrix.min():.4f}, {qfi_matrix.max():.4f}]")
    print(f"Matrix shape: {qfi_matrix.shape} ({N_qubits} qubits)")