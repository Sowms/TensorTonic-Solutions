import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    p = [min(1-eps, max(eps,y_p)) for y_p in y_pred]
    ans = []
    for i, y_t in enumerate(y_true):
        ans.append(-(y_t*math.log(p[i]) + (1-y_t)*math.log(1-p[i])))
    return ans
        