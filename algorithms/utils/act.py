from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    Supports Discrete, Box, MultiBinary, MultiDiscrete, and Tuple combinations
    (e.g., Box+Box or Box+Discrete).
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        # Backward-compat flags
        self.mixed_action = False
        self.multi_discrete = False
        self.mujoco_box = False

        self.action_type = action_space.__class__.__name__
        self.is_tuple = self.action_type in ("Tuple", "tuple")

        if not self.is_tuple:
            # Single-head cases
            if self.action_type == "Discrete":
                action_dim = action_space.n
                self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
            elif self.action_type == "Box":
                self.mujoco_box = True
                action_dim = int(action_space.shape[0])
                self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
            elif self.action_type == "MultiBinary":
                action_dim = int(action_space.shape[0])
                self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
            elif self.action_type == "MultiDiscrete":
                self.multi_discrete = True
                try:
                    nvec = action_space.nvec
                except AttributeError:
                    nvec = action_space.high - action_space.low + 1
                self.action_outs = nn.ModuleList([
                    Categorical(inputs_dim, int(n), use_orthogonal, gain) for n in nvec
                ])
            else:
                # Fallback: treat as tuple-like if it supports iteration
                self.is_tuple = True

        if self.is_tuple:
            # Gymnasium Tuple keeps subspaces in .spaces; gym.Tuple may be iterable
            subspaces = getattr(action_space, "spaces", list(action_space))
            self.heads = nn.ModuleList()
            self.head_info = []  # [{"type": name, "dim": int}]
            for sub in subspaces:
                typ = sub.__class__.__name__
                if typ == "Box":
                    dim = int(sub.shape[0])
                    self.heads.append(DiagGaussian(inputs_dim, dim, use_orthogonal, gain))
                    self.head_info.append({"type": "Box", "dim": dim})
                elif typ == "Discrete":
                    self.heads.append(Categorical(inputs_dim, int(sub.n), use_orthogonal, gain))
                    self.head_info.append({"type": "Discrete", "dim": 1})
                elif typ == "MultiBinary":
                    dim = int(sub.shape[0])
                    self.heads.append(Bernoulli(inputs_dim, dim, use_orthogonal, gain))
                    self.head_info.append({"type": "MultiBinary", "dim": dim})
                elif typ == "MultiDiscrete":
                    # Expand into multiple categorical heads
                    for n in sub.nvec.tolist():
                        self.heads.append(Categorical(inputs_dim, int(n), use_orthogonal, gain))
                        self.head_info.append({"type": "Discrete", "dim": 1})
                else:
                    raise NotImplementedError(f"Unsupported subspace type in Tuple: {typ}")
            self._split_sizes = [info["dim"] for info in self.head_info]

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor or list[Tensor]) available action masks for discrete heads.
        :param deterministic: (bool) whether to return mode instead of sampling.
        """
        # Numerical safety for inputs feeding linear/probability heads
        if torch.is_tensor(x) and torch.is_floating_point(x):
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        if self.is_tuple:
            actions, logprob_terms = [], []
            # allow list/tuple of masks, one per discrete head
            avail_list = None
            if available_actions is not None:
                if isinstance(available_actions, (list, tuple)):
                    avail_list = list(available_actions)
                else:
                    avail_list = [available_actions]
            di = 0  # mask index for discrete heads
            for head, info in zip(self.heads, self.head_info):
                if info["type"] == "Discrete":
                    mask = None
                    if avail_list is not None and di < len(avail_list):
                        mask = avail_list[di]
                        di += 1
                    dist = head(x, mask)
                    a = dist.mode() if deterministic else dist.sample()
                    actions.append(a.float())
                    # expand discrete logprob to its declared dim (=1)
                    lp = dist.log_probs(a)  # (B,1)
                    logprob_terms.append(lp.repeat(1, info["dim"]))
                else:
                    dist = head(x)
                    a = dist.mode() if deterministic else dist.sample()
                    actions.append(a)
                    # expand continuous logprob to per-dimension entries with equal share
                    lp_sum = dist.log_probs(a)  # (B,1), summed over dims
                    dim = max(1, int(info["dim"]))
                    lp_per = lp_sum / dim
                    logprob_terms.append(lp_per.repeat(1, dim))
            actions = torch.cat(actions, dim=-1)
            # concatenate expanded per-head logprobs to match sum of dims
            action_log_probs = torch.cat(logprob_terms, dim=-1)  # (B, sum_dims)
            return actions, action_log_probs

        if self.multi_discrete:
            actions, logprob_terms = [], []
            for head in self.action_outs:
                dist = head(x)
                a = dist.mode() if deterministic else dist.sample()
                actions.append(a)
                logprob_terms.append(dist.log_probs(a))
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(logprob_terms, dim=-1)
            return actions, action_log_probs

        if self.mujoco_box:
            dist = self.action_out(x)
            actions = dist.mode() if deterministic else dist.sample()
            return actions, dist.log_probs(actions)

        # default single-head (Discrete or MultiBinary)
        dist = self.action_out(x, available_actions) if self.action_type == "Discrete" else self.action_out(x)
        actions = dist.mode() if deterministic else dist.sample()
        return actions, dist.log_probs(actions)

    def get_probs(self, x, available_actions=None):
        """Compute action probabilities from inputs (for discrete heads) or means (for continuous heads)."""
        if self.is_tuple:
            out = []
            avail_list = None
            if available_actions is not None:
                if isinstance(available_actions, (list, tuple)):
                    avail_list = list(available_actions)
                else:
                    avail_list = [available_actions]
            di = 0
            for head, info in zip(self.heads, self.head_info):
                if info["type"] == "Discrete":
                    mask = None
                    if avail_list is not None and di < len(avail_list):
                        mask = avail_list[di]
                        di += 1
                    dist = head(x, mask)
                    out.append(dist.probs)
                else:
                    dist = head(x)
                    out.append(dist.mean)
            return torch.cat(out, dim=-1)

        if self.multi_discrete:
            return torch.cat([h(x).probs for h in self.action_outs], dim=-1)

        if self.action_type == "Discrete":
            return self.action_out(x, available_actions).probs
        return self.action_out(x).probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        Returns (action_log_probs, dist_entropy).
        """
        # Numerical safety for inputs feeding linear/probability heads
        if torch.is_tensor(x) and torch.is_floating_point(x):
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        if self.is_tuple:
            parts = torch.split(action, self._split_sizes, dim=-1)
            logprob_terms, ent_terms = [], []
            avail_list = None
            if available_actions is not None:
                if isinstance(available_actions, (list, tuple)):
                    avail_list = list(available_actions)
                else:
                    avail_list = [available_actions]
            di = 0
            for head, info, act_part in zip(self.heads, self.head_info, parts):
                if info["type"] == "Discrete":
                    mask = None
                    if avail_list is not None and di < len(avail_list):
                        mask = avail_list[di]
                        di += 1
                    dist = head(x, mask)
                    act_idx = act_part.long()
                else:
                    dist = head(x)
                    act_idx = act_part
                lp = dist.log_probs(act_idx)
                if active_masks is not None:
                    ent = (dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    ent = dist.entropy().mean()
                # expand per-head lp to per-dimension entries
                if info["type"] == "Discrete":
                    logprob_terms.append(lp.repeat(1, info["dim"]))
                else:
                    dim = max(1, int(info["dim"]))
                    logprob_terms.append((lp / dim).repeat(1, dim))
                ent_terms.append(ent)
            # concatenate to match sum of dims
            action_log_probs = torch.cat(logprob_terms, dim=-1)
            dist_entropy = sum(ent_terms) / len(ent_terms)
            return action_log_probs, dist_entropy

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs, dist_entropy = [], []
            for head, act in zip(self.action_outs, action):
                dist = head(x)
                action_log_probs.append(dist.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(dist.entropy().mean())
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = sum(dist_entropy) / len(dist_entropy)
            return action_log_probs, dist_entropy

        if self.mujoco_box:
            dist = self.action_out(x)
            action_log_probs = dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = dist.entropy().mean()
            return action_log_probs, dist_entropy

        dist = self.action_out(x, available_actions) if self.action_type == "Discrete" else self.action_out(x)
        action_log_probs = dist.log_probs(action)
        if active_masks is not None:
            dist_entropy = (dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = dist.entropy().mean()
        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        """
        TRPO variant: returns (action_log_probs, dist_entropy, action_mu, action_std, all_probs)
        action_mu/std are concatenated for continuous heads; all_probs is logits for discrete heads.
        """
        if self.is_tuple:
            parts = torch.split(action, self._split_sizes, dim=-1)
            logprob_terms, ent_terms = [], []
            mu_list, std_list, logits_list = [], [], []
            avail_list = None
            if available_actions is not None:
                if isinstance(available_actions, (list, tuple)):
                    avail_list = list(available_actions)
                else:
                    avail_list = [available_actions]
            di = 0
            for head, info, act_part in zip(self.heads, self.head_info, parts):
                if info["type"] == "Discrete":
                    mask = None
                    if avail_list is not None and di < len(avail_list):
                        mask = avail_list[di]
                        di += 1
                    dist = head(x, mask)
                    logits_list.append(dist.logits)
                    act_idx = act_part.long()
                else:
                    dist = head(x)
                    mu_list.append(dist.mean)
                    std_list.append(dist.stddev)
                    act_idx = act_part
                lp = dist.log_probs(act_idx)
                if active_masks is not None:
                    if info["type"] == "Discrete":
                        ent = (dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
                    else:
                        ent = (dist.entropy() * active_masks).sum() / active_masks.sum()
                else:
                    ent = dist.entropy().mean()
                logprob_terms.append(lp)
                ent_terms.append(ent)
            action_log_probs = torch.cat(logprob_terms, dim=-1)
            dist_entropy = sum(ent_terms) / len(ent_terms)
            action_mu = torch.cat(mu_list, dim=-1) if len(mu_list) > 0 else None
            action_std = torch.cat(std_list, dim=-1) if len(std_list) > 0 else None
            all_probs = torch.cat(logits_list, dim=-1) if len(logits_list) > 0 else None
            return action_log_probs, dist_entropy, action_mu, action_std, all_probs

        # fall back to original behavior for non-tuple spaces
        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs, dist_entropy = [], []
            mu_collector, std_collector, probs_collector = [], [], []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                mu = getattr(action_logit, "mean", None)
                std = getattr(action_logit, "stddev", None)
                action_log_probs.append(action_logit.log_probs(act))
                if mu is not None:
                    mu_collector.append(mu)
                if std is not None:
                    std_collector.append(std)
                if hasattr(action_logit, "logits"):
                    probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector, -1) if len(mu_collector) > 0 else None
            action_std = torch.cat(std_collector, -1) if len(std_collector) > 0 else None
            all_probs = torch.cat(probs_collector, -1) if len(probs_collector) > 0 else None
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        else:
            action_logits = self.action_out(x, available_actions) if self.action_type == "Discrete" else self.action_out(x)
            action_mu = getattr(action_logits, "mean", None)
            action_std = getattr(action_logits, "stddev", None)
            action_log_probs = action_logits.log_probs(action)
            all_probs = action_logits.logits if hasattr(action_logits, "logits") else None
            if active_masks is not None:
                if self.action_type == "Discrete":
                    dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy() * active_masks).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
