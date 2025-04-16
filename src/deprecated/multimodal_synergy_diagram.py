import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
import matplotlib.gridspec as gridspec

# Create figure and axis
plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# Define time points
t = np.linspace(0, 10, 500)

# Generate example signals with synergistic areas
np.random.seed(42)

# Two input modalities
X1 = 0.8 * np.sin(t) + 0.2 * np.random.randn(len(t))
X2 = 0.8 * np.cos(1.5 * t) + 0.2 * np.random.randn(len(t))

# Target variable - default additive relationship
Y = 0.4 * X1 + 0.4 * X2 + 0.2 * np.random.randn(len(t))

# Synergy areas (where Y depends on X1*X2)
synergy_areas = [(100, 150), (250, 300), (400, 450)]
for start, end in synergy_areas:
    Y[start:end] = 0.1 * X1[start:end] + 0.1 * X2[start:end] + 0.6 * X1[start:end] * X2[start:end] + 0.2 * np.random.randn(end-start)

# Plot original signals
ax1 = plt.subplot(gs[0, :])
ax1.plot(t, X1, 'g-', label='Modality X1')
ax1.plot(t, X2, 'b-', label='Modality X2')
ax1.plot(t, Y, 'r-', label='Target Y')

# Highlight synergy areas
for start, end in synergy_areas:
    ax1.axvspan(t[start], t[end], color='yellow', alpha=0.3)

ax1.set_title('Original Multimodal Time Series Data', fontsize=14)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Signal Value', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Sliding window PID analysis diagram
ax2 = plt.subplot(gs[1, :])

# Create simulated PID components
redundancy = np.zeros_like(t)
unique_x1 = np.zeros_like(t)
unique_x2 = np.zeros_like(t)
synergy = np.zeros_like(t)

# Default values - low synergy, moderate redundancy and unique
redundancy += 0.2 + 0.05 * np.random.randn(len(t))
unique_x1 += 0.15 + 0.05 * np.random.randn(len(t))
unique_x2 += 0.15 + 0.05 * np.random.randn(len(t))
synergy += 0.1 + 0.05 * np.random.randn(len(t))

# High synergy in synergy areas
for start, end in synergy_areas:
    synergy[start:end] = 0.5 + 0.1 * np.random.randn(end-start)
    redundancy[start:end] = 0.15 + 0.05 * np.random.randn(end-start)
    unique_x1[start:end] = 0.1 + 0.05 * np.random.randn(end-start)
    unique_x2[start:end] = 0.1 + 0.05 * np.random.randn(end-start)

# Plot PID components
ax2.plot(t, redundancy, 'b-', label='Redundancy')
ax2.plot(t, unique_x1, 'g-', label='Unique X1')
ax2.plot(t, unique_x2, 'c-', label='Unique X2')
ax2.plot(t, synergy, 'm-', linewidth=2, label='Synergy')

# Add threshold line
threshold = 0.3
ax2.axhline(y=threshold, color='r', linestyle='--', label='Synergy Threshold')

# Highlight detected segments
detected_segments = []
for start, end in synergy_areas:
    # Add some error to make it realistic
    error = np.random.randint(-10, 10)
    detected_start = max(0, start + error)
    error = np.random.randint(-10, 10)
    detected_end = min(len(t)-1, end + error)
    detected_segments.append((detected_start, detected_end))
    
for start, end in detected_segments:
    ax2.axvspan(t[start], t[end], color='orange', alpha=0.3)

ax2.set_title('Temporal PID Analysis', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Information (bits)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Validation with models diagram
ax3 = plt.subplot(gs[2, 0])

# Model performance data
models = ['X1', 'X2', 'X1+X2']
synergy_perf = [0.62, 0.65, 0.88]  # High synergy regions
non_synergy_perf = [0.70, 0.72, 0.78]  # Regular regions

# Bar plot
x_pos = np.arange(len(models))
width = 0.35

ax3.bar(x_pos - width/2, synergy_perf, width, label='High Synergy Regions', color='orange')
ax3.bar(x_pos + width/2, non_synergy_perf, width, label='Regular Regions', color='gray')

# Add labels
ax3.set_xlabel('Model', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Model Performance Comparison', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add accuracy values on bars
for i, v in enumerate(synergy_perf):
    ax3.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
for i, v in enumerate(non_synergy_perf):
    ax3.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')

# Flowchart-style diagram on the right
ax4 = plt.subplot(gs[2, 1])
ax4.axis('off')

# Draw the process flow
y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
labels = [
    "1. Multimodal Time Series",
    "2. Sliding Window PID Analysis",
    "3. Detect High Synergy Segments",
    "4. Train Models on Segments",
    "5. Validate & Visualize"
]

for i, (y, label) in enumerate(zip(y_positions, labels)):
    ax4.add_patch(Rectangle((0.1, y-0.06), 0.8, 0.12, 
                            facecolor='lightblue', edgecolor='black', alpha=0.8))
    ax4.text(0.5, y, label, ha='center', va='center', fontsize=12)
    
    # Add arrows between steps
    if i < len(y_positions) - 1:
        arrow = FancyArrowPatch((0.5, y-0.06), (0.5, y_positions[i+1]+0.06),
                                arrowstyle='->', mutation_scale=20, color='black')
        ax4.add_patch(arrow)

ax4.set_title('Multimodal Synergy Detection Process', fontsize=14)

# Add overall caption
plt.figtext(0.5, 0.01, "Multimodal Synergy Detection Framework: Identifying time segments where combining modalities\n"
                       "provides significantly more information than any single modality alone",
             ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

# Add legend for synergy areas
legend_elements = [
    Patch(facecolor='yellow', alpha=0.3, label='True Synergy Regions'),
    Patch(facecolor='orange', alpha=0.3, label='Detected Synergy Regions')
]
ax1.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# Save the figure
try:
    plt.savefig('results/multimodal_synergy_diagram.png', dpi=300, bbox_inches='tight')
    print("Diagram saved to results/multimodal_synergy_diagram.png")
except Exception as e:
    print(f"Error saving diagram: {e}")
    
plt.savefig('multimodal_synergy_diagram.png', dpi=300, bbox_inches='tight')
print("Diagram saved to multimodal_synergy_diagram.png")

plt.close() 