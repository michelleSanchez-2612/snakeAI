import os
import matplotlib.pyplot as plt

def plot_metrics(episode_scores, mean_scores, record_scores, episode_lengths, episode_number, model_name="Deep_Q_Learning"):
    
    # Create directory for metrics if it doesn't exist
    metrics_folder = f"metrics_{model_name}"
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)
    
    plt.figure(figsize=(12, 8))

    # Subplot 1: Max/Record Score per Episode
    plt.subplot(2, 2, 1)
    plt.plot(episode_scores, label="Score per Episode", color="green", marker="o", linestyle="--")
    plt.plot(record_scores, label="Record", color="red", linestyle="-")
    plt.title("Max/Record Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    # Subplot 2: Episode Length (Number of Steps)
    plt.subplot(2, 2, 2)
    plt.plot(episode_lengths, label="Episode Length", color="purple")
    plt.title("Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    # Subplot 3: Learning Curve (Mean Score)
    plt.subplot(2, 2, 3)
    plt.plot(mean_scores, label="Mean Score", color="orange")
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Mean Score")
    plt.legend()

    plt.tight_layout()
    file_name = f"metrics_{episode_number}.png"
    file_path = os.path.join(metrics_folder, file_name)
    plt.savefig(file_path, dpi=150)
    plt.close()
