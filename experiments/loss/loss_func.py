import torch
import torch.nn as nn


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y, c=c,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check

    # print("S_padded.shape", S_padded.shape, S_padded)


    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss

def risk_from_logits(h):
    """Risk of discrete-time hazard logits (B, K) -> (B,): -sum of the survival curve, the same score as
    ModelManager.calculate_risk (the one the c-index is computed on), but differentiable.
    A head with one output (B, 1) is a Cox model: its output is the log hazard ratio, used as it is."""
    if h.shape[-1] == 1:
        return h[..., 0]
    return -torch.cumprod(1 - torch.sigmoid(h), dim=-1).sum(dim=-1)


def cox_ph_loss(risk, time, c, reduction='mean'):
    """Negative Cox partial log-likelihood (Breslow ties) of a set of patients.
    risk (B,) log hazard ratios; time (B,) follow-up time; c (B,) censorship (1 = censored).
    The risk set of an event at time t_i is every patient with t_j >= t_i. 'mean' divides by the number of events.
    Without events in the set the loss is 0 (with a graph, so that backward still runs)."""
    risk = risk.reshape(-1)
    time = time.reshape(-1).to(risk.device, torch.float32)
    event = (1 - c.reshape(-1).to(risk.device, torch.float32))
    if event.sum() == 0:
        return risk.sum() * 0.0
    at_risk = time.unsqueeze(0) >= time.unsqueeze(1)                       # [i, j]: j in the risk set of i
    log_denominator = torch.logsumexp(risk.unsqueeze(0).masked_fill(~at_risk, float('-inf')), dim=1)
    neg_l = -((risk - log_denominator) * event).sum()
    if reduction == 'mean':
        return neg_l / event.sum()
    if reduction == 'sum':
        return neg_l
    raise ValueError("Bad input for reduction: {}".format(reduction))


class CoxPHLoss(nn.Module):
    """Cox partial likelihood on the risk of the head (risk_from_logits), so the score trained is the one
    evaluated by the c-index. With a one-output head (model output_dim 1) the risk is the log hazard ratio
    (a Cox model); with the K-bin head it is -sum of the survival curve, in (-K, 0).

    risk_set:
      "set"     the likelihood of risk_set_size consecutive training patients (graphs kept), one optimizer step
                per set (the 2026-10-09 clinical_encoding runs)
      "cohort"  the likelihood of the WHOLE training set: ModelManager keeps the latest risk of every training
                patient (no graph); each patient's term is the cohort likelihood with its own risk live, so the
                gradient of a batch is an estimate of the full-cohort gradient (exact when the stored risks are
                current), and the optimizer steps every real_batch_size patients, as with NLLSurvLoss
    Validation / test always use the whole set. Time = follow-up in days (original_event_time), not the bin.

    alpha / eps: accepted only because the grid versions are merged into the yaml's NLLSurvLoss kwargs; unused.
    The K-bin curves of a model trained with this loss are not fitted as distributions; with output_dim 1 the
    survival curves come from a Breslow baseline hazard fitted on the training set (ModelManager)."""
    def __init__(self, risk_set_size=16, risk_set="set", reduction='mean', alpha=0.0, eps=1e-7):
        super().__init__()
        if risk_set not in ("set", "cohort"):
            raise ValueError(f"CoxPHLoss: risk_set {risk_set}: set | cohort")
        if risk_set == "set" and risk_set_size < 2:
            raise ValueError("CoxPHLoss: risk_set_size must be >= 2 (one patient has no partial likelihood)")
        self.risk_set = risk_set
        self.risk_set_size = int(risk_set_size)
        self.reduction = reduction

    def from_risk(self, risk, time, c):
        return cox_ph_loss(risk, time, c, reduction=self.reduction)

    def __call__(self, h, y, t, c):
        """Same signature as NLLSurvLoss: h (B,K) logits, y (B,1) bin (used only when t is None), t (B,1)
        follow-up time, c (B,1) censorship."""
        return self.from_risk(risk_from_logits(h), y if t is None else t, c)
