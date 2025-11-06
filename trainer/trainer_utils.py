"""训练脚本公用的辅助函数集合。"""

import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM


def is_main_process():
    """判断当前进程是否负责日志打印或保存。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """仅在主进程输出日志，避免重复信息。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """计算带常数下限的 Cosine 学习率。"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    """若检测到 DDP 环境变量则初始化 ``torch.distributed``。"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """同时设定 Python、NumPy 与 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model',
                   vision_model_path='../model/vision_model/clip-vit-base-patch16',
                   save_dir='../out', device='cuda', freeze_llm=False):
    """构建 MiniMind-VLM 并可选加载预训练权重。

    Args:
        vlm_config (VLMConfig): 模型结构配置。
        from_weight (str): 要加载的权重前缀，``'none'`` 表示跳过。
        tokenizer_path (str): tokenizer 保存目录。
        vision_model_path (str): CLIP 视觉模型路径。
        save_dir (str): 权重所在目录。
        device (str): 模型放置设备。
        freeze_llm (bool): 是否冻结 LLM，仅训练视觉桥接层。

    Returns:
        Tuple[MiniMindVLM, transformers.PreTrainedTokenizer, CLIPProcessor]:
        分别为模型、tokenizer 与图像预处理器。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)

    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # Pretrain阶段：冻结除 vision_proj 外的所有参数
    if freeze_llm:
        for name, param in model.named_parameters():
            if 'vision_proj' not in name:
                param.requires_grad = False

    Logger(f'所加载VLM Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """保存或加载 MiniMind-VLM 训练过程的 checkpoint。

    Args:
        vlm_config (VLMConfig): 用于命名 checkpoint 的配置。
        weight (str): 权重前缀。
        model (Optional[nn.Module]): 训练时传入模型执行保存；``None`` 表示加载模式。
        optimizer (Optional[Optimizer]): 需要持久化的优化器对象。
        epoch (int): 当前 epoch。
        step (int): 当前 step。
        wandb: 可选的 W&B 实例。
        save_dir (str): checkpoint 存储目录。
        **kwargs: 其他需要保存的对象，若具有 ``state_dict`` 会被调用。

    Returns:
        Optional[dict]: 在加载模式下返回恢复字典，否则 ``None``。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 移除vision_encoder参数（不需要保存，因为是预训练的）
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    """包装采样器，用于跳过前若干批次。"""

    def __init__(self, sampler, batch_size, skip_batches=0):
        """记录底层采样器与跳过配置。

        Args:
            sampler (Sampler): 原始采样器。
            batch_size (int): 单个 batch 包含的样本数。
            skip_batches (int): 需要跳过的批次数。
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """迭代生成 batch，忽略前 ``skip_batches`` 个。"""
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """返回实际可用的 batch 数量。"""
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

