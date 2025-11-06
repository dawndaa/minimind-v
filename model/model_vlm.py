import os

import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    """MiniMind 多模态配置，额外记录视觉提示 token。"""

    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            **kwargs,
    ):
        """设置图像 patch 占位符的默认文本形式。

        Args:
            image_special_token (str): Prompt 中替换 `<image>` 的特定字符序列。
            image_ids (List): tokenizer 编码后对应的 token id 列表。
            **kwargs: 透传给 ``MiniMindConfig`` 的其他参数。
        """
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)

class VisionProj(nn.Module):
    """线性映射头，将 CLIP patch 特征对齐到 LLM 隐藏维度。"""

    def __init__(self, ve_hidden_size=768, hidden_size=512):
        """保存输入输出维度并构建线性层。

        Args:
            ve_hidden_size (int): 视觉编码器输出维度，例如 CLIP ViT 为 768。
            hidden_size (int): 语言模型隐藏维度。
        """
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        """将一批 CLIP 特征映射到语言空间。

        Args:
            image_encoders (torch.Tensor): patch 级视觉特征 ``(bsz, patches, ve_hidden_size)``。

        Returns:
            torch.Tensor: 线性映射后的特征 ``(bsz, patches, hidden_size)``。
        """
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    """MiniMind 视觉-语言模型，将 CLIP 特征注入语言流。"""

    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        """加载语言模型与 CLIP 编码器，并初始化投影层。

        Args:
            params (VLMConfig, optional): 多模态配置，默认构造。.
            vision_model_path (str): 本地 CLIP checkpoint 路径。
        """
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        """加载 CLIP 主干并冻结参数以提升推理效率。

        Args:
            model_path (str): CLIP checkpoint 路径。

        Returns:
            Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]: 成功时返回模型与处理器，否则 ``(None, None)``。
        """
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        """将任意 PIL 图像转换为 CLIP 归一化 Tensor。

        Args:
            image (PIL.Image.Image): 输入图片。
            processor (CLIPProcessor): CLIP 预处理器。

        Returns:
            torch.Tensor: 归一化后的像素 ``(1, 3, H, W)``。
        """
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        """从冻结 CLIP 模型提取 patch-level embedding。

        Args:
            image_tensors (torch.Tensor): ``image2tensor`` 的输出。
            vision_model (CLIPModel): 视觉编码器。

        Returns:
            torch.Tensor: 去掉 CLS 后的 patch 表示。
        """
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """用视觉 patch embedding 替换 `<image>` 占位片段。

        Args:
            tokens (torch.Tensor): Prompt token 序列。
            h (torch.Tensor): 语言嵌入序列。
            vision_tensors (Optional[torch.Tensor]): 视觉特征。
            seqlen (int): 输出长度上限。

        Returns:
            torch.Tensor: 注入视觉特征后的嵌入序列。
        """
        def find_indices(tokens, image_ids):
            """在 prompt 中定位 patch 占位区间。"""
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # ``vision_proj`` already stores a sequence of
                            # patch embeddings; we splice them into the
                            # language sequence in place of the placeholders.
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[:seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        """在语言 forward 过程中融合视觉特征。

        Args:
            input_ids (Optional[torch.Tensor]): 文本 token 序列。
            attention_mask (Optional[torch.Tensor]): 注意力掩码。
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): 历史 KV。
            use_cache (bool): 是否返回新的 KV。
            logits_to_keep (Union[int, torch.Tensor]): 输出 logits 的切片范围。
            pixel_values (Optional[torch.FloatTensor]): 图像像素 ``(bsz, num_img, 3, H, W)``。
            **args: 透传到语言模型的其它参数。

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: 包含融合结果。
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT
