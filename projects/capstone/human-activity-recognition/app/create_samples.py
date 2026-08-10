import os
import pandas as pd
import numpy as np

# X_test is your test tensor with shape (2947, 128, 9)
# y_test is one-hot encoded
# CLASS_NAMES = [...]

CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

output_dir = "../app/sample_data"
os.makedirs(output_dir, exist_ok=True)

saved = set()

for i in range(len(X_test)):

    label = CLASS_NAMES[np.argmax(y_test[i])]

    filename = label.lower().replace(" ", "_") + ".csv"

    if filename in saved:
        continue

    df = pd.DataFrame(
        X_test[i],
        columns=[
            "Body Acc X",
            "Body Acc Y",
            "Body Acc Z",
            "Body Gyro X",
            "Body Gyro Y",
            "Body Gyro Z",
            "Total Acc X",
            "Total Acc Y",
            "Total Acc Z",
        ],
    )

    df.to_csv(
        os.path.join(output_dir, filename),
        index=False,
        header=False,
    )

    saved.add(filename)

    if len(saved) == 6:
        break

print("Sample CSV files created successfully!")