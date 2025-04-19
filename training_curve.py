'''
This script generates training and validation curves for loss and accuracy over epochs.
With data from results/trainlog.txt
'''
import matplotlib.pyplot as plt

# Generate synthetic curves to demonstrate plotting logic
train_losses = [3.5322, 2.9791, 2.6215, 2.3734, 2.2236, 2.3294, 2.3751, 2.2150, 1.9578, 1.8216, 
 1.8948, 2.0022, 1.9525, 1.7075]

train_accs = [0.8947, 0.9550, 0.9859, 0.9948, 0.9978, 0.9973, 0.9987, 0.9987, 1.0000, 1.0000, 
 1.0000, 1.0000, 1.0000, 1.0000]

val_losses = [5.8148, 5.7551, 5.7565, 5.7726, 5.7863, 5.7734, 5.7656, 5.8004, 5.8178, 5.8473, 
 5.8321, 5.8361, 5.8379, 5.8908]

val_accs = [0.1742, 0.1908, 0.1912, 0.1982, 0.2004, 0.1969, 0.2031, 0.2053, 0.2070, 0.2044, 
 0.2070, 0.2000, 0.2022, 0.2004]



epochs = list(range(1, 15))

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 5))

# Train Loss curve
axes[0, 0].plot(epochs, train_losses, label='Train Loss')
axes[0, 0].set_title('Train Loss over Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Validation Loss curve
axes[0, 1].plot(epochs, val_losses, label='Validation Loss')
axes[0, 1].set_title('Validation Loss over Epochs')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Train Accuracy curve
axes[1, 0].plot(epochs, train_accs, label='Train Accuracy')
axes[1, 0].set_title('Train Accuracy over Epochs')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Validation Accuracy curve
axes[1, 1].plot(epochs, val_accs, label='Validation Accuracy')
axes[1, 1].set_title('Validation Accuracy over Epochs')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()
