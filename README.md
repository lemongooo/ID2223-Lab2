# ID2223 Lab2

# Task2: Improve pipeline scalability and model performance
## Data-Centric Improvement Approach

### Dataset Evolution
- Original dataset: **FineTome-100k**, a subset of arcee-ai/The-Tome
- Enhanced dataset: Expanded to 200K samples from the same parent dataset (arcee-ai/The-Tome)

### Implementation Details
```python
dataset = load_dataset("arcee-ai/The-Tome", split="train").shuffle(seed=42).select(range(200000))
```

### Dataset Comparison
- Original: 100K samples from FineTome-100k
- New: 200K randomly sampled conversations from The-Tome
- Both share the same data source, ensuring consistency in data quality and style

This approach doubles the training data while maintaining data source consistency, potentially leading to better model performance.