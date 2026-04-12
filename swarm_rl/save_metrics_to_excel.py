import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

def extract_metrics_from_tensorboard(logdir):
    """Extract metrics from TensorBoard logs."""
    metrics = {}
    for subdir in os.listdir(logdir):
        subdir_path = os.path.join(logdir, subdir)
        if os.path.isdir(subdir_path):
            event_files = [f for f in os.listdir(subdir_path) if "tfevents" in f]
            if event_files:
                event_file = os.path.join(subdir_path, event_files[0])
                event_acc = EventAccumulator(event_file)
                event_acc.Reload()
                
                metrics[subdir] = {}
                for tag in event_acc.Tags()["scalars"]:
                    scalar_events = event_acc.Scalars(tag)
                    steps = [e.step for e in scalar_events]
                    values = [e.value for e in scalar_events]
                    metrics[subdir][tag] = {"steps": steps, "values": values}
    return metrics

def save_metrics_to_excel(logdir, output):
    """Save extracted metrics to an Excel file."""
    metrics = extract_metrics_from_tensorboard(logdir)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for experiment, tags in metrics.items():
            for tag, data in tags.items():
                df = pd.DataFrame({"Step": data["steps"], "Value": data["values"]})
                sheet_name = f"{experiment[:10]}_{tag[:20]}"[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"📊 Metrics saved to: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save TensorBoard metrics to Excel.")
    parser.add_argument("--logdir", type=str, required=True, help="Path to the TensorBoard log directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the Excel file.")
    args = parser.parse_args()
    
    save_metrics_to_excel(args.logdir, args.output)
