import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from model.model_vlm import MiniMindVLM
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLMDataset(Dataset):
    """配对对话文本与预处理图片的自定义数据集。"""

    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):
        """加载数据元信息并准备 tokenizer 相关常量。

        Args:
            jsonl_path (str): 训练样本 JSONL 路径。
            images_path (str): 图像文件所在目录。
            tokenizer: 支持 chat template 的 tokenizer。
            preprocess: 图像预处理函数，通常为 CLIPProcessor。
            max_length (int): 序列最大长度。
            image_special_token (str): Prompt 中的 `<image>` 占位符文本。
        """

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        """返回样本数量（对话-图像对的个数）。"""
        return len(self.samples)

    def load_data(self, path):
        """读取 JSONL 文件，每行对应一个训练样本。

        Args:
            path (str): JSONL 文件路径。

        Returns:
            List[dict]: 原始样本列表。
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """使用 chat template 构造对话 prompt 并插入图像占位符。

        Args:
            conversations (List[dict]): 含 ``role`` 与 ``content`` 的对话列表。

        Returns:
            str: 展开占位符后的完整 prompt 字符串。
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """生成 loss mask，仅计算 assistant 回复段的损失。

        Args:
            input_ids (List[int]): prompt 编码后的 token id 序列。

        Returns:
            List[int]: 与序列等长的 0/1 掩码。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        """编码指定样本的对话并加载对应图像。

        Args:
            index (int): 样本索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ``X``: 输入序列。
            - ``Y``: 预测目标序列。
            - ``loss_mask``: 损失掩码。
            - ``image_tensors``: 图像 patch 张量。
        """
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return X, Y, loss_mask, image_tensors
