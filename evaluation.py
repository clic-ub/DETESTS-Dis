import os
import re

import pandas as pd
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


def evaluate(pred, gold, evaluate_as_hard=False):
    task = int(gold.split(".")[0].split("_")[-2][-1:])
    label_type = gold.split(".")[0].split("_")[-1]

    for fname in [pred, gold]:
        if label_type == "hard" and not os.path.isfile(fname):
            soft_f = "_".join(fname[:-5].split("_")[:-1]) + "_soft.json"
            if os.path.isfile(soft_f):
                soft_to_hard(soft_f)
            else:
                raise FileNotFoundError(f"{soft_to_hard(fname)} was not found.")

    if task == 2:
        params = {
            PyEvALLUtils.PARAM_HIERARCHY: {
                "Stereotype": ["Implicit", "Explicit"],
                "NoStereotype": [],
            }
        }
        if label_type == "soft":
            metrics = ["ICMSoft", "ICMSoftNorm"]
        elif label_type == "hard":
            metrics = ["ICM", "ICMNorm", "Precision", "Recall"]
        else:
            raise ValueError(f'Invalid label_type ({label_type}): Must be either "soft" or "hard".')
    elif task == 1:
        params = {}
        if label_type == "soft":
            metrics = ["CrossEntropy"]
        elif label_type == "hard":
            metrics = ["FMeasure", "Precision", "Recall"]
        else:
            raise ValueError(f'Invalid label_type ({label_type}): Must be either "soft" or "hard".')
    else:
        raise ValueError(
            f"Invalid task: {task}. DETESTS-Dis Shared class only has two available tasks ({{1, 2}})."
        )

    test = PyEvALLEvaluation()
    if isinstance(pred, list):
        report = test.evaluate_lst(pred, gold, metrics, **params)
    else:
        report = test.evaluate(pred, gold, metrics, **params)

    if task == 1 and label_type == "hard":
        return {
            "F1": report.report["metrics"]["FMeasure"]["results"]["test_cases"][0]["classes"][
                "Stereotype"
            ],
            "Precision": report.report["metrics"]["Precision"]["results"]["test_cases"][0][
                "classes"
            ]["Stereotype"],
            "Recall": report.report["metrics"]["Recall"]["results"]["test_cases"][0]["classes"][
                "Stereotype"
            ],
        }
    elif task == 2 and label_type == "hard":
        return {
            "ICM": report.report["metrics"]["ICM"]["results"]["test_cases"][0]["average"],
            "ICM Norm": report.report["metrics"]["ICMNorm"]["results"]["test_cases"][0]["average"],
            "PrecisionImplicit": report.report["metrics"]["Precision"]["results"]["test_cases"][0][
                "classes"
            ]["Implicit"],
            "PrecisionExplicit": report.report["metrics"]["Precision"]["results"]["test_cases"][0][
                "classes"
            ]["Explicit"],
            "RecallImplicit": report.report["metrics"]["Recall"]["results"]["test_cases"][0][
                "classes"
            ]["Implicit"],
            "RecallExplicit": report.report["metrics"]["Recall"]["results"]["test_cases"][0][
                "classes"
            ]["Explicit"],
        }
    elif task == 1 and label_type == "soft":
        return {
            "Cross Entropy": report.report["metrics"]["CrossEntropy"]["results"]["test_cases"][0][
                "average"
            ]
        }
    else:
        return {
            "ICM Soft": report.report["metrics"]["ICMSoft"]["results"]["test_cases"][0]["average"],
            "ICM Soft Norm": report.report["metrics"]["ICMSoftNorm"]["results"]["test_cases"][0][
                "average"
            ],
        }


def test_example(name, team_name="example_pred"):
    if name == "t1_soft":
        return f"data/sample/{team_name}_t1_soft.json", "data/sample/example_gold_t1_soft.json"
    if name == "t1_hard":
        return f"data/sample/{team_name}_t1_hard.json", "data/sample/example_gold_t1_hard.json"
    if name == "t2_soft":
        return f"data/sample/{team_name}_t2_soft.json", "data/sample/example_gold_t2_soft.json"
    if name == "t2_hard":
        return f"data/sample/{team_name}_t2_hard.json", "data/sample/example_gold_t2_hard.json"


def soft_to_hard(fname):
    def transform(d):
        ds = pd.Series(d)
        if "Implicit" in ds and (ds.Implicit + ds.Explicit > ds.NoStereotype):
            return ds[["Implicit", "Explicit"]].idxmax()

        return pd.Series(d).idxmax()

    print(f"Transforming Soft into Hard Labels for file {fname}")
    df = pd.read_json(fname)
    df["value"] = df["value"].apply(transform)
    df.to_json(re.sub("soft", "soft2hard", fname), orient="records", indent=4)


def main():
    pred, gold = test_example("t1_hard")
    print("Task 1 Hard Labels", evaluate(pred, gold))

    pred, gold = test_example("t1_soft")
    print("Task 1 Soft Labels", evaluate(pred, gold))

    pred, gold = test_example("t2_soft")
    print("Task 2 Hard Labels", evaluate(pred, gold))

    pred, gold = test_example("t2_hard")
    print("Task 2 Soft Labels", evaluate(pred, gold))


if __name__ == "__main__":
    main()
