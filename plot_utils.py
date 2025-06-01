
import matplotlib.pyplot as plt

def plot_smoothed_loss(loss_history, output_path="smoothed_loss.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, linewidth=1)
    plt.title("Smoothed Training Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Smoothed Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
