import pandas as pd
import os

# Step 1: Create initial sample data (version v1)
data = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [85, 90, 95]
}
df = pd.DataFrame(data)

# Step 2: Add a new row for data version v2
new_row_v2 = {"id": 4, "name": "David", "score": 88}
df = pd.concat([df, pd.DataFrame([new_row_v2])], ignore_index=True)

# Step 3: Add a new row for data version v3
new_row_v3 = {"id": 5, "name": "Eve", "score": 92}
df = pd.concat([df, pd.DataFrame([new_row_v3])], ignore_index=True)

# Step 4: Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)

# Step 5: Save the DataFrame to a CSV file
df.to_csv("data/sample_data.csv", index=False)

print("Data saved to data/sample_data.csv")
