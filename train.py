
"""Train and evaluate three crop classification models."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "Crop_recommendation.csv"
)

MODEL_DIR = (
    BASE_DIR / "backend" / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. DATASET CONFIGURATION
# --------------------------------------------------

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

TARGET = "label"


# --------------------------------------------------
# 3. MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("\n==========================================")
    print("       CROPSENSE MODEL TRAINING")
    print("==========================================")


    # ----------------------------------------------
    # STEP 1: LOAD DATASET
    # ----------------------------------------------

    print("\n[1/10] Loading dataset...")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = FEATURES + [TARGET]

    missing_columns = [
        column for column in required
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required].isnull().any().any():

        raise ValueError(
            "Dataset contains missing values. "
            "Clean them before training."
        )

    if not all(
        pd.api.types.is_numeric_dtype(df[column])
        for column in FEATURES
    ):

        raise ValueError(
            "All seven input features must be numeric."
        )

    X = df[FEATURES].astype(float)

    y = df[TARGET].astype(str)

    classes = sorted(y.unique().tolist())

    if len(classes) < 2:

        raise ValueError(
            "At least two crop classes are required."
        )

    if y.value_counts().min() < 5:

        raise ValueError(
            "Each crop class must have at least "
            "5 records for 5-fold cross-validation."
        )

    print("Dataset loaded successfully.")

    print("Total records:", len(df))

    print("Input features:", len(FEATURES))

    print("Crop classes:", len(classes))


    # ----------------------------------------------
    # STEP 2: TRAIN-TEST SPLIT
    # ----------------------------------------------

    print("\n[2/10] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(X_train))

    print("Testing samples:", len(X_test))


    # ----------------------------------------------
    # STEP 3: DEFINE MODELS
    # ----------------------------------------------

    print("\n[3/10] Initializing models...")

    models = {

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "KNN": Pipeline([

            ("scaler", StandardScaler()),

            ("classifier", KNeighborsClassifier(
                n_neighbors=5
            ))

        ])

    }

    print("Three models initialized.")


    # ----------------------------------------------
    # STEP 4: CONFIGURE CROSS-VALIDATION
    # ----------------------------------------------

    print("\n[4/10] Configuring cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {

        "accuracy": "accuracy",

        "precision_macro": "precision_macro",

        "recall_macro": "recall_macro",

        "f1_macro": "f1_macro"

    }

    print("Method: 5-Fold Stratified Cross-Validation")

    print("Execution: Sequential (n_jobs=1)")


    # ----------------------------------------------
    # STEP 5: INITIALIZE RESULT CONTAINERS
    # ----------------------------------------------

    results = {}

    fitted_models = {}

    best_name = None

    best_cv_f1 = -1.0

    best_cv_accuracy = -1.0


    # ----------------------------------------------
    # STEP 6: TRAIN AND EVALUATE EACH MODEL
    # ----------------------------------------------

    print("\n[5/10] Evaluating classification models...")

    for name, model in models.items():

        print("\n------------------------------------------")

        print("Evaluating:", name)

        print("------------------------------------------")


        # Cross-validation on training data
        cv_result = cross_validate(

            model,

            X_train,

            y_train,

            cv=cv,

            scoring=scoring,

            n_jobs=1,

            return_train_score=False

        )


        # Fit model on complete training split
        start = time.perf_counter()

        model.fit(
            X_train,
            y_train
        )

        fit_ms = (
            time.perf_counter() - start
        ) * 1000


        # Predict test-set labels
        predictions = model.predict(X_test)


        # ------------------------------------------
        # STEP 7: CALCULATE TEST METRICS
        # ------------------------------------------

        test_metrics = {

            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions
                )
            ),

            "precision_macro": float(
                precision_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),

            "recall_macro": float(
                recall_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),

            "f1_macro": float(
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            )

        }


        # ------------------------------------------
        # STEP 8: CALCULATE CV METRICS
        # ------------------------------------------

        cv_metrics = {

            "cv_accuracy_mean": float(
                np.mean(
                    cv_result["test_accuracy"]
                )
            ),

            "cv_accuracy_std": float(
                np.std(
                    cv_result["test_accuracy"]
                )
            ),

            "cv_accuracy_folds": [
                float(x)
                for x in cv_result["test_accuracy"]
            ],

            "cv_precision_macro_mean": float(
                np.mean(
                    cv_result["test_precision_macro"]
                )
            ),

            "cv_recall_macro_mean": float(
                np.mean(
                    cv_result["test_recall_macro"]
                )
            ),

            "cv_f1_macro_mean": float(
                np.mean(
                    cv_result["test_f1_macro"]
                )
            ),

            "cv_f1_macro_std": float(
                np.std(
                    cv_result["test_f1_macro"]
                )
            ),

            "cv_f1_macro_folds": [
                float(x)
                for x in cv_result["test_f1_macro"]
            ]

        }


        # ------------------------------------------
        # STEP 9: STORE EVALUATION RESULTS
        # ------------------------------------------

        results[name] = {

            **test_metrics,

            **cv_metrics,

            "fit_time_ms": float(fit_ms),

            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=classes
            ).tolist(),

            "classification_report": classification_report(
                y_test,
                predictions,
                labels=classes,
                output_dict=True,
                zero_division=0
            )

        }


        # Store fitted model
        fitted_models[name] = model


        # Save individual model
        filename = (
            name.lower().replace(" ", "_") + ".pkl"
        )

        joblib.dump(
            model,
            MODEL_DIR / filename
        )


        # Display evaluation results
        print("\nTest Results:")

        print(
            "Accuracy:",
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            "Precision:",
            f"{test_metrics['precision_macro']:.4f}"
        )

        print(
            "Recall:",
            f"{test_metrics['recall_macro']:.4f}"
        )

        print(
            "Macro F1:",
            f"{test_metrics['f1_macro']:.4f}"
        )


        print("\nCross-Validation Results:")

        print(
            "Fold-wise Accuracy:",
            [
                round(x, 4)
                for x in cv_metrics["cv_accuracy_folds"]
            ]
        )

        print(
            "Fold-wise Macro F1:",
            [
                round(x, 4)
                for x in cv_metrics["cv_f1_macro_folds"]
            ]
        )

        print(
            "Mean CV Accuracy:",
            f"{cv_metrics['cv_accuracy_mean']:.4f}"
        )

        print(
            "Mean CV Macro F1:",
            f"{cv_metrics['cv_f1_macro_mean']:.4f}"
        )


        # ------------------------------------------
        # STEP 10: SELECT BEST MODEL
        # ------------------------------------------

        candidate_f1 = (
            cv_metrics["cv_f1_macro_mean"]
        )

        candidate_accuracy = (
            cv_metrics["cv_accuracy_mean"]
        )

        if (

            candidate_f1 > best_cv_f1

            or (

                np.isclose(
                    candidate_f1,
                    best_cv_f1
                )

                and candidate_accuracy > best_cv_accuracy

            )

        ):

            best_name = name

            best_cv_f1 = candidate_f1

            best_cv_accuracy = candidate_accuracy


    # --------------------------------------------------
    # 7. SAVE SELECTED BEST MODEL
    # --------------------------------------------------

    print("\n==========================================")

    print("Selecting best model...")

    best_model = fitted_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )

    print("Selected model:", best_name)


    # --------------------------------------------------
    # 8. GENERATE METADATA
    # --------------------------------------------------

    metadata = {

        "project_name": "CropSense",

        "dataset_name": "Crop Recommendation Dataset",

        "dataset_rows": int(len(df)),

        "feature_names": FEATURES,

        "class_names": classes,

        "class_distribution": {
            str(label): int(count)
            for label, count in y.value_counts().items()
        },

        "training_samples": int(len(X_train)),

        "testing_samples": int(len(X_test)),

        "test_size": 0.20,

        "random_state": RANDOM_STATE,

        "cross_validation": {

            "method": "StratifiedKFold",

            "folds": 5,

            "shuffle": True,

            "random_state": RANDOM_STATE,

            "execution": "Sequential"

        },

        "selection_metric": (
            "Mean cross-validation macro F1-score "
            "(tie-break: CV accuracy)"
        ),

        "best_model": best_name,

        "best_cv_f1_macro": float(best_cv_f1),

        "best_cv_accuracy": float(best_cv_accuracy),

        "models": results

    }


    # --------------------------------------------------
    # 9. SAVE METADATA JSON
    # --------------------------------------------------

    metadata_path = (
        MODEL_DIR / "model_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("Metadata saved successfully.")


    # --------------------------------------------------
    # 10. FINAL SUMMARY
    # --------------------------------------------------

    print("\n==========================================")

    print("       TRAINING COMPLETED")

    print("==========================================")

    print("\nMODEL COMPARISON")

    print("------------------------------------------")

    print(
        f"{'Model':<18}"
        f"{'CV Accuracy':<18}"
        f"{'CV Macro F1':<18}"
        f"{'Test Accuracy':<18}"
    )

    print("------------------------------------------")

    for name, result in results.items():

        print(
            f"{name:<18}"
            f"{result['cv_accuracy_mean']:<18.4f}"
            f"{result['cv_f1_macro_mean']:<18.4f}"
            f"{result['accuracy']:<18.4f}"
        )

    print("------------------------------------------")

    print("\nBEST MODEL:", best_name)

    print(
        "Best CV Macro F1:",
        f"{best_cv_f1:.4f}"
    )

    print(
        "Best CV Accuracy:",
        f"{best_cv_accuracy:.4f}"
    )

    print("\nGENERATED FILES")

    print("------------------------------------------")

    print("1. naive_bayes.pkl")

    print("2. random_forest.pkl")

    print("3. knn.pkl")

    print("4. best_model.pkl")

    print("5. model_metadata.json")

    print("\nSaved inside:", MODEL_DIR)

    print("\nCropSense training completed successfully!")

    print("==========================================")


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()
