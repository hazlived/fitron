import pandas as pd

from pip_model import FITRONModel


df = pd.DataFrame(
    {
        "income": [50000, 20000, 78000, 30000, 60000, 47000, 71000, 25000],
        "risk": [0.3, 0.8, 0.2, 0.7, 0.4, 0.5, 0.3, 0.9],
        "credit_score": [650, 500, 770, 560, 700, 640, 730, 520],
        "employment_years": [2, 1, 10, 2, 6, 4, 9, 1],
        "status": ["approve", "reject", "approve", "reject", "approve", "approve", "approve", "reject"],
    }
)

model = FITRONModel(iterations=5, random_state=42)
result = model.fit(df, target="status", target_map={"reject": 0, "approve": 1})

print("Best index:", result.best_index)
print("Best score:", result.best_score)
print("Train/Test accuracy:", result.train_accuracy, result.test_accuracy)
print("Explanation:")
for item in result.explanation:
    print(" -", item)
