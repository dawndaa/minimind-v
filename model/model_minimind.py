# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """MiniMind 语言模型配置对象。

    中文简介以便快速理解：该配置类集中描述 MiniMind LLM 与多模态封装
    （MiniMind-VLM）共享的超参数，便于在训练、推理之间复用统一设置；所有
    字段都提供默认值，使得初始化时无需显式传入长参数表，但仍能完整呈现模
    型结构。

    English reference: configuration container for MiniMind language model that
    exposes architectural hyper-parameters shared by both standalone LLM and
    multimodal wrapper.
    """

    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        """初始化配置并保存所有关键参数。

        Args:
            dropout (float): Dropout 概率，控制 Attention/FFN 的随机失活。
            bos_token_id (int): ``<bos>`` 特殊 token id。
            eos_token_id (int): ``<eos>`` 特殊 token id。
            hidden_act (str): 前馈网络激活函数名称，例如 ``silu``。
            hidden_size (int): Transformer 隐状态维度。
            intermediate_size (int): 前馈层中间维度，``None`` 时按 8/3 倍自动推断。
            max_position_embeddings (int): 位置编码支持的最大序列长度。
            num_attention_heads (int): Multi-Head Attention 的头数。
            num_hidden_layers (int): Transformer block 数量。
            num_key_value_heads (int): KV cache 使用的头数，``None`` 时等于 ``num_attention_heads``。
            vocab_size (int): 词表大小。
            rms_norm_eps (float): RMSNorm 中的数值稳定常数 ``epsilon``。
            rope_theta (float): Rotary Position Embedding 的频率基数。
            inference_rope_scaling (bool): 是否启用 YaRN 长序列外推策略。
            flash_attn (bool): 是否在可用时使用 FlashAttention。
            use_moe (bool): 是否启用 MoE FeedForward。
            num_experts_per_tok (int): 每个 token 选择的专家数 ``top_k``。
            n_routed_experts (int): MoE 中可被路由的专家总数。
            n_shared_experts (int): 始终参与的 shared expert 数量。
            scoring_func (str): Gate logits 归一化方式，默认 ``softmax``。
            aux_loss_alpha (float): 辅助损失系数。
            seq_aux (bool): 辅助损失是否按序列统计而非按 batch。
            norm_topk_prob (bool): 是否对 top-k 概率进行归一化。
            **kwargs: 透传给 ``PretrainedConfig`` 的其他字段。

        Notes:
            - Method 仅保存参数，不做 heavy computation。
            - 当 ``inference_rope_scaling`` 为真时，会提前构造 YaRN 所需的配置字典。
        """
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """MiniMind 使用的 RMSNorm（Root Mean Square Normalization）。

    English reference: RMS layer normalisation variant that scales activations
    by their root mean square without subtracting the mean.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """构造归一化层并初始化缩放参数。

        Args:
            dim (int): 隐藏维度大小，对应需要归一化的 feature size。
            eps (float): 数值稳定项 ``epsilon``，避免除零。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """按最后一维的 RMS 对张量进行归一化。

        Args:
            x (torch.Tensor): 任意形状的输入张量。

        Returns:
            torch.Tensor: 经过 RMS 归一化的结果，仍保持 ``x`` 的 dtype。
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """对输入张量执行 RMSNorm 并施加可学习缩放。

        Args:
            x (torch.Tensor): 待归一化的隐藏状态。

        Returns:
            torch.Tensor: 归一化并缩放后的输出。
        """
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """预计算 Rotary Embedding 所需的 cos/sin 位置编码。

    Args:
        dim (int): 单头的隐藏维度（偶数），会生成 ``dim/2`` 个频率。
        end (int): 预生成的最大 ``position``，通常等于 ``max_position_embeddings``。
        rope_base (float): RoPE 基数 ``theta``，影响频率间距。
        rope_scaling (Optional[dict]): YaRN 风格的外推配置，``None`` 表示不缩放。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(freqs_cos, freqs_sin)``，每个张量形状为
        ``(end, dim)``，用于 ``apply_rotary_pos_emb``。

    Notes:
        - English reference: helper optionally applies YaRN scaling for long context extrapolation.
    """
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            # YaRN rescales the lower frequency bands while shrinking the high
            # frequencies so that the same projection matrix works for longer
            # contexts.
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """对 Query/Key 应用旋转位置编码。

    Args:
        q (torch.Tensor): Query 张量，形状 ``(bsz, seq_len, n_heads, head_dim)``。
        k (torch.Tensor): Key 张量，同上。
        cos (torch.Tensor): 预计算的 cos 项。
        sin (torch.Tensor): 预计算的 sin 项。
        position_ids (Optional[torch.Tensor]): 自定义位置索引，默认顺序位置。
        unsqueeze_dim (int): cos/sin 扩张维度，默认 ``1`` 与 FlashAttention 对齐。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 注入位置编码后的 ``(q_embed, k_embed)``。
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 KV 头以匹配 Query 头数。

    Args:
        x (torch.Tensor): ``(bsz, seq_len, kv_heads, head_dim)`` 形状的 key/value 张量。
        n_rep (int): 每个 KV 头需要复制的次数 ``n_query // n_kv``。

    Returns:
        torch.Tensor: 展平复制后的张量，形状 ``(bsz, seq_len, n_query, head_dim)``。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """多头 Self-Attention 层，可选启用 FlashAttention。"""

    def __init__(self, args: MiniMindConfig):
        """初始化 QKV 投影矩阵及注意力配置。

        Args:
            args (MiniMindConfig): 与当前层相关的配置对象。
        """
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """执行带 RoPE 的自注意力，并支持 KV cache。

        Args:
            x (torch.Tensor): 输入隐状态 ``(bsz, seq_len, hidden)``。
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): ``(cos, sin)`` 旋转位置编码切片。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): 过往 ``(k, v)``。
            use_cache (bool): 是否返回新的 ``past_key_value``。
            attention_mask (Optional[torch.Tensor]): 序列掩码 ``(bsz, seq_len)``。

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            - ``output``: 注意力输出 ``(bsz, seq_len, hidden)``。
            - ``past_kv``: 若 ``use_cache``，返回新的 ``(k, v)``，否则 ``None``。

        Notes:
            English reference: supports both FlashAttention via ``scaled_dot_product_attention``
            and manual softmax fallback.
        """
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """基于 SwiGLU 的前馈网络模块。"""

    def __init__(self, config: MiniMindConfig):
        """根据配置构建门控前馈网络并设置 Dropout。

        Args:
            config (MiniMindConfig): 提供隐藏维度、激活函数等信息。
        """
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """执行门控激活并映射回隐藏维度。

        Args:
            x (torch.Tensor): 输入张量 ``(bsz, seq_len, hidden)``。

        Returns:
            torch.Tensor: 经过 SwiGLU 门控与线性变换后的输出。
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """MoE FeedForward 的 token 级门控网络。"""

    def __init__(self, config: MiniMindConfig):
        """初始化路由权重并存储辅助损失配置。

        Args:
            config (MiniMindConfig): 包含 MoE 相关参数，如专家数、top-k 等。
        """
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用 Kaiming Uniform 初始化门控权重。"""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """执行 top-k 路由并返回辅助损失。

        Args:
            hidden_states (torch.Tensor): 输入隐藏状态 ``(bsz, seq_len, hidden)``。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - ``topk_idx``: 每个 token 选中的专家索引。
            - ``topk_weight``: 对应的归一化权重。
            - ``aux_loss``: 训练阶段的负载均衡辅助损失。
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """可切换的 Mixture-of-Experts 前馈层。"""

    def __init__(self, config: MiniMindConfig):
        """构建专家池、门控及可选共享专家。

        Args:
            config (MiniMindConfig): MoE 架构配置。
        """
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """完成 token 路由并聚合 MoE 辅助损失。

        Args:
            x (torch.Tensor): 输入隐藏状态 ``(bsz, seq_len, hidden)``。

        Returns:
            torch.Tensor: 汇总专家输出后的张量，形状与输入一致。
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """在推理阶段执行高效的 MoE 路由。

        Args:
            x (torch.Tensor): 展平的 token 表示 ``(tokens, hidden)``。
            flat_expert_indices (torch.Tensor): 每个 token 对应的专家索引。
            flat_expert_weights (torch.Tensor): 归一化的专家权重。

        Returns:
            torch.Tensor: 与 ``x`` 同形状的聚合结果。
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """由 Self-Attention 与 FFN 组成的 Transformer 基本模块。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):
        """搭建单层 Transformer 子结构。

        Args:
            layer_id (int): 当前层索引，用于调试或缓存命名。
            config (MiniMindConfig): 模型配置。
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """执行注意力与前馈并应用残差连接。

        Args:
            hidden_states (torch.Tensor): 输入隐藏状态。
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): 位置编码切片。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): 过往 KV。
            use_cache (bool): 是否返回新的 KV。
            attention_mask (Optional[torch.Tensor]): 序列掩码。

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 处理后的隐藏状态与 KV。
        """
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """MiniMind 语言模型骨干，负责生成 decoder 隐状态。"""

    def __init__(self, config: MiniMindConfig):
        """初始化词嵌入、Transformer 层与 RoPE 缓存。

        Args:
            config (MiniMindConfig): 模型结构配置。
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """计算一批 token 序列的隐藏状态。

        Args:
            input_ids (Optional[torch.Tensor]): token id 序列 ``(bsz, seq_len)``。
            attention_mask (Optional[torch.Tensor]): 自回归掩码。
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): 历史 KV。
            use_cache (bool): 是否收集新的 KV。
            **kwargs: 兼容 ``GenerationMixin`` 的额外参数。

        Returns:
            Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
            - 最终隐藏状态。
            - 每层的 KV。
            - MoE 辅助损失标量。

        Notes:
            English reference: signature mirrors ``PreTrainedModel`` forward for generation API.
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniMind 自回归语言模型封装，输出 logits 与 KV cache。"""

    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        """实例化 Transformer 主体并创建共享输出头。

        Args:
            config (MiniMindConfig, optional): 若为空则使用默认配置。
        """
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """返回 logits、隐藏状态与 KV cache。

        Args:
            input_ids (Optional[torch.Tensor]): 输入 token 序列。
            attention_mask (Optional[torch.Tensor]): attention mask。
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): 历史 KV。
            use_cache (bool): 是否缓存 KV。
            logits_to_keep (Union[int, torch.Tensor]): 仅保留最新 logits 的切片范围。
            **args: 透传给 ``MiniMindModel`` 的额外参数。

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: 含 logits、隐藏状态、KV、MoE 损失。
        """
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
