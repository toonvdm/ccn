import torch
import torch.nn.functional as F
from torch.distributions.kl import register_kl
from torch.distributions.utils import logits_to_probs

"""
  We use custom classes for distributions that are also torch.Tensor instances
  This is a bit hacky, but allows to handle them nicely in TensorDict
"""


def Categorical(logits):
    # this is already a Categorical distribution, return
    if isinstance(logits, InnerCategorical):
        return logits

    # call type only once to make sure .__class__ is equals for objects
    if not hasattr(Categorical, "cls"):
        cls = type('Categorical', (InnerCategorical, logits.__class__), {})
        setattr(Categorical, "cls", cls)

    logits.__class__ = Categorical.cls
    return logits


def Uniform(batch_size, num_classes):
    return Categorical(torch.ones(batch_size, num_classes))


class InnerCategorical:
    """ A custom Categorical distribution that uses
    Gumbel Softmax reparameterization trick for sampling

    Sampled values are now one hot encoded vectors instead of numbers
    """

    @property
    def logits(self):
        return self

    @property
    def probs(self):
        return logits_to_probs(self.logits)

    def gumbel_sample(self, shape):
        eps = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(eps + 1e-20) + 1e-20)

    def sample(self, no_samples=1, temperature=0.1, hard=False):
        sample_shape = self.logits.shape
        if no_samples > 1:
            sample_shape = [no_samples] + list(sample_shape)
        y = self.logits + self.gumbel_sample(sample_shape)
        y = F.softmax(y / temperature, dim=-1)

        if not hard:
            return y

        _, indices = y.max(-1)
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, indices.view(-1, 1), 1)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)

    def to(self, device):
        moved = self.__class__.__bases__[1].to(self, device)
        if isinstance(moved, InnerCategorical):
            return moved
        return Categorical(moved)

    def __getitem__(self, index):
        item = self.__class__.__bases__[1].__getitem__(self, index)
        return Categorical(item)


@register_kl(InnerCategorical, InnerCategorical)
def kl_cat_cat(p, q):
    log_ratio = q.logits - p.logits
    KL = -torch.sum(p.probs * log_ratio, dim=-1)
    return KL
