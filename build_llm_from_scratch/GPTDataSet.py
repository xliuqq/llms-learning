import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataSetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        '''
        Docstring for __init__
        
        :param txt: 输入文本内容
        :param tokenizer: 词元编码器
        :param max_length: 上下文长度
        :param stride: 滑动窗口大小
        '''
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        token_size = len(token_ids)

        # 使用滑动窗口，将文本划分为长度为 max_length 的（重叠）序列
        for i in range(0, token_size - max_length, stride):
            # 最后的批次数据可能不足 max_length 个 token
            input_chunk = token_ids[i:i + max_length]
            # target 右移以为表示预测
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
                      num_workers=0, encoding_name="gpt2"):
    '''
    Docstring for create_dataloader
    
    :param txt: 输入文本内容
    :param tokenizer: 词元编码器
    :param max_length: 上下文长度
    :param stride: 滑动窗口大小
    :param batch_size: 批次大小
    '''
    tokenizer = tiktoken.get_encoding(encoding_name)
    dataset = GPTDataSetV1(txt, tokenizer, max_length, stride)
    # 如果 drop_last 为 True，则最后一个批次的数据数量可能会小于 batch_size 会被丢弃
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers)
    
    return dataloader