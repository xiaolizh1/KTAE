from typing import Any, Callable, Optional, Sized, Union, Dict
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
logger = logging.getLogger(__name__)
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def compute_Cohen(a, b):
    a = torch.clamp(a, min=0, max=1)
    b = torch.clamp(b, min=0, max=1)
    h = torch.asin(torch.sqrt(a)) - torch.asin(torch.sqrt(b))
    return h


def fisher_exact_test(input_tensor):
    a, b, c, d = torch.unbind(input_tensor, dim=1)
    R1 = a + b
    R2 = c + d
    C1 = a + c
    n = R1 + R2
    k_min = torch.max(torch.tensor(0, dtype=torch.float32), R1 + C1 - n)
    k_max = torch.min(R1, C1)
    max_k = torch.max(k_max)
    k_values = torch.arange(max_k + 1, dtype=torch.float32, device=input_tensor.device)
    k_grid = torch.tile(k_values, (input_tensor.size(0), 1))  # [batch, max_k+1]
    mask = (k_grid >= k_min.unsqueeze(1)) & (k_grid <= k_max.unsqueeze(1))
    C1_grid = C1.unsqueeze(1).expand_as(k_grid)
    R1_grid = R1.unsqueeze(1).expand_as(k_grid)
    n_grid = n.unsqueeze(1).expand_as(k_grid)
    k = k_grid
    def log_comb(n, k):
        return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)

    log_p_k = log_comb(C1_grid, k) + log_comb(n_grid - C1_grid, R1_grid - k) - log_comb(n_grid, R1_grid)
    p_k = torch.exp(log_p_k) * mask.float()
    log_p_obs = log_comb(C1, a) + log_comb(n - C1, R1 - a) - log_comb(n, R1)
    p_obs = torch.exp(log_p_obs)
    p_values = torch.sum(torch.where((p_k <= p_obs.unsqueeze(1)) & mask, p_k, 0.), dim=1)

    return p_values


def calculate_ig(table_tensor):
    t_a = table_tensor[:, 0]
    t_b = table_tensor[:, 1]
    t_c = table_tensor[:, 2]
    t_d = table_tensor[:, 3]
    t_n = t_a + t_b + t_c + t_d
    def safe_log(x):
        return torch.where(x > 0, torch.log2(x), torch.tensor(0.0, dtype=torch.float32, device=x.device))
    def entropy(p):
        return -p * safe_log(p)
    p_correct = (t_a + t_c) / t_n
    p_error = (t_b + t_d) / t_n
    h_y = entropy(p_correct) + entropy(p_error)
    def conditional_entropy(x1, x2):
        total = x1 + x2
        return torch.where(total > 0,
                           entropy(x1 / total) + entropy(x2 / total),
                           torch.tensor(0.0, dtype=torch.float32, device=x1.device))

    h_y_x = ((t_a + t_b)/t_n) * conditional_entropy(t_a, t_b) + \
            ((t_c + t_d)/t_n) * conditional_entropy(t_c, t_d)
    ig = h_y - h_y_x
    ig[t_n == 0] = 0
    return ig


def compute_tf(correct_tensor, correct_mask, wrong_tensor, wrong_mask, appear_table_tensor):
    k1 = 2.0
    b = 0.5
    correct_tensor_unmasked = correct_tensor[correct_mask == 1]
    wrong_tensor_unmasked = wrong_tensor[wrong_mask == 1]
    total_correct = correct_tensor_unmasked.numel()
    total_wrong = wrong_tensor_unmasked.numel()
    avg_len = (total_correct + total_wrong) / 2
    corrcet_k = k1 * (1 - b + b * (total_correct / avg_len))
    wrong_k = k1 * (1 - b + b * (total_wrong / avg_len))
    tf_tensor = torch.zeros((appear_table_tensor.size(0), 2), dtype=torch.float32, device=appear_table_tensor.device)
    correct_hist = torch.bincount(correct_tensor_unmasked, minlength=appear_table_tensor.size(0))
    wrong_hist = torch.bincount(wrong_tensor_unmasked, minlength=appear_table_tensor.size(0))
    tf_tensor = torch.stack(
        [(correct_hist.float() / total_correct) * (k1 + 1) / ((correct_hist.float() / total_correct) + corrcet_k),
         (wrong_hist.float() / total_wrong) * (k1 + 1) / ((wrong_hist.float() / total_wrong) + wrong_k)],
        dim=1).to(dtype=torch.float32, device=appear_table_tensor.device)

    return tf_tensor


class ComputeKeyTokens:

    def __init__(self, alpha, beta_ig, gamma_tf, top, bottom, responses_ids, mask, rewards, max_token_num):
        self.top = top
        self.bottom = bottom
        self.alpha = alpha
        self.beta_ig = beta_ig
        self.gamma_tf = gamma_tf
        self.responses_ids = responses_ids
        self.mask = mask
        self.rewards = rewards
        self.max_token_num = max_token_num

    def get_distribution(self, tensor, mask):
        appear_distribution = {}
        m, n = tensor.shape
        device = tensor.device
        row_indices = torch.arange(m)[:, None].expand(m, n)  # [[0,0,0], [1,1,1], ...]
        row_indices = row_indices.to(device)
        flat_values = tensor.flatten()
        flat_rows = row_indices.flatten()
        flat_mask = mask.flatten().to(device)
        masked_values = flat_values[flat_mask == 1]
        masked_rows = flat_rows[flat_mask == 1]
        unique_values, inverse_indices = torch.unique(masked_values, return_inverse=True)
        for i, val in enumerate(unique_values.tolist()):
            current_rows = masked_rows[inverse_indices == i]
            unique_rows = torch.unique(current_rows)
            appear_distribution[val] = len(unique_rows)
        return appear_distribution

    def compute_key_tokens(self, correct_tensor, wrong_tensor, correct_mask, wrong_mask):
        key_token_result = torch.zeros(self.max_token_num + 1, dtype=torch.float32)
        if correct_tensor.size(0) == 0 or wrong_tensor.size(0) == 0:
            return key_token_result
        correct_dist = self.get_distribution(correct_tensor, correct_mask)
        wrong_dist = self.get_distribution(wrong_tensor, wrong_mask)
        appear_table_tensor = torch.zeros(
            (self.max_token_num + 1, 4),
            dtype=torch.float32,
            device=correct_tensor.device
        )
        tokenid_set = set(self.responses_ids.flatten().tolist())
        for tokenid in tokenid_set:
            correct_count = correct_dist.get(tokenid, 0)
            wrong_count = wrong_dist.get(tokenid, 0)
            total_correct = correct_tensor.size(0)
            total_wrong = wrong_tensor.size(0)
            if correct_count + wrong_count != 0:
                appear_table_tensor[tokenid] = torch.tensor(
                    [correct_count, wrong_count, total_correct - correct_count, total_wrong - wrong_count],
                    dtype=torch.float32, device=correct_tensor.device)
        ig_val = calculate_ig(appear_table_tensor) if self.beta_ig else torch.zeros(
            self.max_token_num + 1, dtype=torch.float32, device=appear_table_tensor.device)
        if self.alpha:
            fisher_pvalue = fisher_exact_test(appear_table_tensor)
            format_fisher_pvalue = torch.where(
                fisher_pvalue != 1.0, torch.exp(-2 * fisher_pvalue),
                torch.tensor(0.0, dtype=torch.float32, device=appear_table_tensor.device))
        else:
            format_fisher_pvalue = torch.zeros(self.max_token_num + 1,
                                               dtype=torch.float32,
                                               device=appear_table_tensor.device)
        if self.gamma_tf:
            tf_tensor = compute_tf(correct_tensor, correct_mask, wrong_tensor, wrong_mask, appear_table_tensor)
            x=compute_Cohen(appear_table_tensor[:, 0] / (appear_table_tensor[:, 0] + appear_table_tensor[:, 2] + 1e-10), appear_table_tensor[:, 1] / (appear_table_tensor[:, 1] + appear_table_tensor[:, 3] + 1e-10))
            x+=tf_tensor[:,0]/(tf_tensor[:,1]+1e-5)-tf_tensor[:,1]/(tf_tensor[:,0]+1e-5)
        else:
            x=compute_Cohen(appear_table_tensor[:, 0] / (appear_table_tensor[:, 0] + appear_table_tensor[:, 2] + 1e-10), appear_table_tensor[:, 1] / (appear_table_tensor[:, 1] + appear_table_tensor[:, 3] + 1e-10))
        key_token_result = ((self.alpha * format_fisher_pvalue + self.beta_ig * ig_val) *x) / (self.alpha + self.beta_ig)
        key_token_result=(torch.sigmoid(key_token_result)-0.5)*2*self.top
        key_token_result[151643] = 0.
        return key_token_result

    def get_key_tokens(self):
        correct_outputs = self.responses_ids[self.rewards > 0.0]
        correct_outputs_mask = self.mask[self.rewards > 0.0]
        wrong_outputs = self.responses_ids[self.rewards <= 0.0]
        wrong_outputs_mask = self.mask[self.rewards <= 0.0]
        key_token_result = self.compute_key_tokens(correct_outputs, wrong_outputs, correct_outputs_mask,
                                                   wrong_outputs_mask)
        return key_token_result
