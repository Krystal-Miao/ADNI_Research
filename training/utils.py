import pandas as pd

import torch

def get_class_weights(csv_path, device):

    df = pd.read_csv(csv_path)

    train_subjects = (

        df[df["split"] == "train"][["subject", "label"]]

        .drop_duplicates()

    )

    counts = train_subjects["label"].value_counts().sort_index()

    cn = counts[0]

    ad = counts[1]

    class_weights = torch.tensor(

        [1.0, cn / ad],

        dtype=torch.float32,

        device=device,

    )

    print(f"Training subjects: CN={cn}, AD={ad}")

    print(f"Class weights: {class_weights}")

    return class_weights