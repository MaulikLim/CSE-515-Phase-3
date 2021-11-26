import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List


def _get_fp_misses(y_true: List[int], y_pred: List[int]) -> [int, int, int, int]:
    cnf_mat = confusion_matrix(y_true, y_pred)
    FP = cnf_mat.sum(axis=0) - np.diag(cnf_mat)
    FN = cnf_mat.sum(axis=1) - np.diag(cnf_mat)
    TP = np.diag(cnf_mat)
    TN = cnf_mat.sum() - (FP + FN + TP)
    return TP, TN, FP, FN


def print_matrices(y_true: List[int], y_pred: List[int]) -> None:
    """

    :type y_pred: List
    :type y_true: List
    """
    eps = 1e-5
    TP, TN, FP, FN = _get_fp_misses(y_true, y_pred)
    FPR = FP / (TN + FP + eps)
    misses = FN / (TP + FN + eps)
    with np.printoptions(precision=3, suppress=True, formatter={'float': '{: 0.2f}'.format}):
        print(f"False Positive rate(in percent) for each class:\n {FPR * 100}")
        print(f"Miss rate(in percent) for each class:\n {misses * 100}")
        print(f"Overall accuracy:\n {sum(y_pred == y_true) * 100 / len(y_true):.2f}")
