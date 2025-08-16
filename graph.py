import matplotlib.pyplot as plt
import yaml

# Load YAML data
with open("training_log_refactored.yaml", "r") as f:
    data = yaml.safe_load(f)

train_losses = data["epoch_average_train_losses"]
val_losses = data["epoch_average_val_losses"]
epochs = list(range(1, len(train_losses) + 1))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
plt.plot(epochs, val_losses, label="Validation Loss", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Epochs")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
# plt.show()
plt.savefig("training_validation_loss_plot.png", dpi=300)
