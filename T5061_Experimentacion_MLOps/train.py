import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"
EXPERIMENT = "TC5061_Titanic_LogisticRegression_Comparacion"


def parse_args():
    p = argparse.ArgumentParser(description="Entrena baseline Titanic y registra la corrida en MLflow.")
    p.add_argument("--data-path", default="train.csv")
    p.add_argument("--c", type=float, default=1.0, help="Regularización inversa C (>0).")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    return p.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")
    df = df.dropna(subset=[TARGET])
    return df[FEATURES], df[TARGET].astype(int)


def build_pipeline(c, max_iter, class_weight):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("categorical", categorical, CATEGORICAL)])
    model = LogisticRegression(C=c, max_iter=max_iter,
                               class_weight=None if class_weight == "none" else "balanced",
                               random_state=42)
    return Pipeline([("preprocessing", prep), ("model", model)])


def main():
    args = parse_args()
    if args.c <= 0 or args.max_iter < 1 or not 0 < args.test_size < 1:
        raise ValueError("Usa C > 0, max-iter >= 1 y test-size entre 0 y 1.")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    pipeline = build_pipeline(args.c, args.max_iter, args.class_weight)

    with mlflow.start_run(run_name=f"LR_C{args.c}_iter{args.max_iter}_{args.class_weight}"):
        params = {
            "model": "LogisticRegression", "C": args.c, "max_iter": args.max_iter,
            "class_weight": args.class_weight, "test_size": args.test_size,
            "random_state": args.random_state, "features": ", ".join(FEATURES),
            "dataset_rows": len(X), "train_rows": len(X_train), "test_rows": len(X_test)
        }
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        model = pipeline.named_steps["model"]

        print("C utilizado:", model.C)
        print("Máximo de iteraciones:", model.max_iter)
        print("Iteraciones realizadas:", model.n_iter_)
        print("Clases predichas:", pd.Series(pred).value_counts().to_dict())
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0))
        }
        mlflow.log_metrics(metrics)

        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, pred),
            display_labels=["No sobrevivió", "Sobrevivió"]
        )
        cm_display.plot()
        plt.title("Matriz de confusión - Titanic")
        plt.tight_layout()
        cm_path = artifacts / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        metrics_path = artifacts / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(metrics_path))
        # Pickle evita los problemas de lista de tipos confiables que pueden surgir con skops.
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model",
                                 serialization_format="pickle")
        print(f"Experimento: {EXPERIMENT}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
