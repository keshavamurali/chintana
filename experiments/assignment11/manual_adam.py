"""
A from-scratch, dependency-free reimplementation of the Adam update rule
(Kingma & Ba, 2015), applied one gradient at a time to a single weight.

This mirrors exactly what torch.optim.Adam does per parameter (with
weight_decay=0):

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat_t = m_t / (1 - beta1^t)         (bias correction)
    v_hat_t = v_t / (1 - beta2^t)         (bias correction)
    update_t = lr * m_hat_t / (sqrt(v_hat_t) + eps)
    w_t = w_{t-1} - update_t
"""

from dataclasses import dataclass, field


@dataclass
class AdamStepResult:
    t: int
    grad: float
    m: float
    v: float
    m_hat: float
    v_hat: float
    update: float
    w: float


@dataclass
class AdamState:
    w: float
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    bias_correction: bool = True
    t: int = 0
    m: float = 0.0
    v: float = 0.0
    history: list = field(default_factory=list)

    def step(self, grad: float) -> AdamStepResult:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad

        if self.bias_correction:
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
        else:
            m_hat = self.m
            v_hat = self.v

        update = self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        self.w -= update

        result = AdamStepResult(
            t=self.t, grad=grad, m=self.m, v=self.v,
            m_hat=m_hat, v_hat=v_hat, update=update, w=self.w,
        )
        self.history.append(result)
        return result

    def run(self, grads):
        return [self.step(g) for g in grads]
