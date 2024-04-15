import os
import pandas as pd

from pyevall.utils.utils import PyEvALLUtils
from pyevall.evaluation import PyEvALLEvaluation


def evaluate(pred, gold, evaluate_as_hard=False):
    task = int(gold.split('.')[0].split('_')[-2][-1:])
    label_type = gold.split('.')[0].split('_')[-1]

    for fname in [pred, gold]:
        if label_type == 'hard' and not os.path.isfile(fname):
            soft_f = '_'.join(fname[:-5].split('_')[:-1]) + "_soft.json"
            if os.path.isfile(soft_f):
                soft_to_hard(soft_f)
            else:
                raise FileNotFoundError(f'{soft_to_hard(fname)} was not found.')

    if task == 2:
        params = {PyEvALLUtils.PARAM_HIERARCHY: {"Stereotype": ["Implicit", "Explicit"], "NoStereotype": []}}
        if label_type == 'soft':
            metrics = ["ICMSoft", "ICMSoftNorm"]
        elif label_type == 'hard':
            metrics = ["ICM", "ICMNorm"]
        else:
            raise ValueError(f'Invalid label_type ({label_type}): Must be either "soft" or "hard".')
    elif task == 1:
        params = {}
        if label_type == 'soft':
            metrics = ["CrossEntropy"]
        elif label_type == 'hard':
            metrics = ["FMeasure", "Precision", "Recall"]
        else:
            raise ValueError(f'Invalid label_type ({label_type}): Must be either "soft" or "hard".')
    else:
        raise ValueError(f"Invalid task: {task}. DETESTS-Dis Shared class only has two available tasks ({{1, 2}}).")

    test = PyEvALLEvaluation()
    if isinstance(pred, list):
        report = test.evaluate_lst(pred, gold, metrics, **params)
    else:
        report = test.evaluate(pred, gold, metrics, **params)
    report.print_report()


def test_example(name, team_name='example_pred'):
    if name == 't1_soft':
        return f'data/sample/{team_name}_t1_soft.json', 'data/sample/gt_t1_soft.json'
    if name == 't1_hard':
        return f'data/sample/{team_name}_t1_hard.json', 'data/sample/gt_t1_hard.json'
    if name == 't2_soft':
        return f'data/sample/{team_name}_t2_soft.json', 'data/sample/gt_t2_soft.json'
    if name == 't2_hard':
        return f'data/sample/{team_name}_t2_hard.json', 'data/sample/gt_t2_hard.json'


def soft_to_hard(fname):
    def transform(d):
        return pd.Series(d).idxmax()

    print(f'Transforming Soft into Hard Labels for file {fname}')
    df = pd.read_json(fname)
    df['value'] = df['value'].apply(transform)
    df.to_json('_'.join(fname[:-5].split('_')[:-1]) + "_hard.json", orient='records', indent=4)


if __name__ == "__main__":
    # pred, gold = test_example('t1_soft')
    pred, gold = test_example('t1_hard')
    # pred, gold = test_example('t2_soft')
    # pred, gold = test_example('t2_hard')
    evaluate(pred, gold)

