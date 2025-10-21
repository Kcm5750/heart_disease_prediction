import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.Heart.utils.utils import save_object


def generate_synthetic_dataset(num_rows: int = 1000) -> tuple[pd.DataFrame, np.ndarray]:
    # Columns expected by the app/pipeline
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        'age': rng.integers(29, 78, size=num_rows),
        'sex': rng.integers(0, 2, size=num_rows),
        'cp': rng.integers(0, 4, size=num_rows),
        'trestbps': rng.integers(90, 200, size=num_rows),
        'chol': rng.integers(120, 600, size=num_rows),
        'fbs': rng.integers(0, 2, size=num_rows),
        'restecg': rng.integers(0, 2, size=num_rows),
        'thalach': rng.integers(70, 210, size=num_rows),
        'exang': rng.integers(0, 2, size=num_rows),
        'oldpeak': rng.uniform(0.0, 6.5, size=num_rows),
        'slope': rng.integers(0, 3, size=num_rows),
        'ca': rng.integers(0, 4, size=num_rows),
        'thal': rng.integers(0, 3, size=num_rows),
    }, columns=columns)

    # Create a synthetic binary target with some weak signal
    logits = (
        0.03 * data['age']
        + 0.8 * data['sex']
        + 0.5 * data['cp']
        + 0.01 * data['trestbps']
        + 0.005 * data['chol']
        - 0.02 * data['thalach']
        + 0.6 * data['exang']
        + 0.4 * data['oldpeak']
        - 0.3 * data['slope']
        + 0.2 * data['ca']
        + 0.1 * data['thal']
    )
    probs = 1 / (1 + np.exp(-(logits - logits.mean()) / (logits.std() + 1e-6)))
    y = (probs > 0.5).astype(int).to_numpy()

    return data, y


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ('num_pipeline', num_pipeline, feature_columns)
    ])
    return preprocessor


def main() -> None:
    X, y = generate_synthetic_dataset(num_rows=1500)

    feature_columns = list(X.columns)
    preprocessor = build_preprocessor(feature_columns)

    X_processed = preprocessor.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_processed, y)

    os.makedirs('Artifacts', exist_ok=True)
    save_object(os.path.join('Artifacts', 'Preprocessor.pkl'), preprocessor)
    save_object(os.path.join('Artifacts', 'Model.pkl'), model)


if __name__ == "__main__":
    main()


