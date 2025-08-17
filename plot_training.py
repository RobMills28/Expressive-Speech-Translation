import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_file(filepath):
    train_data = []
    cv_data = []
    # Regex to capture relevant lines and metrics
    train_pattern = re.compile(r".*TRAIN Batch (\d+)/(\d+).*loss ([\d.]+)\sacc ([\d.]+)")
    cv_pattern = re.compile(r".*Epoch (\d+) Step \d+ CV info.*loss ([\d.]+)\sacc ([\d.]+)")

    with open(filepath, 'r') as f:
        for line in f:
            train_match = train_pattern.search(line)
            cv_match = cv_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                # Ignoring batch steps for a simpler epoch-level plot
            elif cv_match:
                epoch = int(cv_match.group(1))
                loss = float(cv_match.group(2))
                acc = float(cv_match.group(3))
                cv_data.append({'epoch': epoch, 'cv_loss': loss, 'cv_acc': acc})
    return cv_data

# Parse both log files
cv_data1 = parse_log_file('training_log.txt')
cv_data2 = parse_log_file('training_log_resumed.txt')

# Combine and sort the data
all_cv_data = sorted(cv_data1 + cv_data2, key=lambda x: x['epoch'])

# Use pandas for easier handling of final epoch values
df = pd.DataFrame(all_cv_data)
# Get the last recorded CV value for each epoch
final_cv_df = df.groupby('epoch').last().reset_index()

# --- We need training data too ---
# Simplified: We'll manually add the final training stats we found earlier
# In a real scenario, you'd parse the final TRAIN batch line for each epoch
train_stats = {
    0: {'train_loss': 3.508, 'train_acc': 0.233},
    1: {'train_loss': 3.251, 'train_acc': 0.287},
    2: {'train_loss': 3.490, 'train_acc': 0.216},
    3: {'train_loss': 3.508, 'train_acc': 0.226},
    4: {'train_loss': 3.581, 'train_acc': 0.216},
    5: {'train_loss': 2.990, 'train_acc': 0.264},
    6: {'train_loss': 2.434, 'train_acc': 0.355},
    7: {'train_loss': 1.453, 'train_acc': 0.603},
    8: {'train_loss': 1.767, 'train_acc': 0.519},
    9: {'train_loss': 1.079, 'train_acc': 0.733}
}
final_cv_df['train_loss'] = final_cv_df['epoch'].map(lambda e: train_stats.get(e, {}).get('train_loss'))
final_cv_df['train_acc'] = final_cv_df['epoch'].map(lambda e: train_stats.get(e, {}).get('train_acc'))


# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Loss
ax1.plot(final_cv_df['epoch'], final_cv_df['train_loss'], 'bo-', label='Training Loss')
ax1.plot(final_cv_df['epoch'], final_cv_df['cv_loss'], 'ro-', label='Validation Loss')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss per Epoch')
ax1.legend()
ax1.grid(True)

# Plot Accuracy
ax2.plot(final_cv_df['epoch'], final_cv_df['train_acc'], 'bo-', label='Training Accuracy')
ax2.plot(final_cv_df['epoch'], final_cv_df['cv_acc'], 'ro-', label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy per Epoch')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
print("Plot saved to training_curves.png")