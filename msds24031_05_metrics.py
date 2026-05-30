import json
from pathlib import Path

def save_metrics():

    metrics = {

        "student_name": "Mubashar Hussain",
        "roll_number": "MSDS24031",
        "seed": 2026,
        "batch_size": 64,
        "simclr_epochs": 50,
        "linear_probe_epochs": 20,
        "finetuning_epochs": 20,
        "learning_rate": 0.0003,
        "temperature": 0.5,

        "supervised_10percent_test_acc": 45.87,
        "random_linear_probe_test_acc": 26.960,
        "simclr_linear_probe_test_acc": 74.660,
        "simclr_finetune_test_acc": 81.150,

        "same_view_similarity_before": 0.992,
        "different_image_similarity_before": 0.988,
        "same_view_similarity_after": 0.905,
        "different_image_similarity_after": 0.462
    }

    Path('results').mkdir(exist_ok=True)

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    save_metrics()