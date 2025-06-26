import json
import torch
import numpy as np

def to_serializable(obj):
    """
    Recursively convert torch tensors, numpy arrays, and other non-serializable
    objects to Python native types (float, list, dict).
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (float, int, str, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return str(obj)  # fallback: convert unknown types to string

def convert_simulation_summary_to_serializable(simulation_summary):
    """
    Converts simulation_summary containing tensors and other complex types into
    a fully JSON-serializable list of dictionaries.
    """
    serializable_summary = []

    for entry in simulation_summary:
        if isinstance(entry, (list, tuple)) and len(entry) == 4:
            theta, phi, rss, cir = entry
            serializable_summary.append({
                "theta": to_serializable(theta),
                "phi": to_serializable(phi),
                "rss": to_serializable(rss),
                "cir": to_serializable(cir)
            })
        else:
            # fallback for malformed entries
            serializable_summary.append(to_serializable(entry))

    return serializable_summary



serializable_data = convert_simulation_summary_to_serializable(simulation_summary)

with open("simulation_summary.json", "w") as f:
    json.dump(serializable_data, f, indent=4)
