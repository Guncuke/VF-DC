# 这个方法将接收原始浮点型embedding和位数bits作为输入，并返回量化后再反量化的tensor。
# 这个quantize_and_dequantize方法将先量化输入的tensor到指定的位数，然后立即反量化回浮点型。这样，你可以直接比较原始tensor和处理后的tensor，以观察量化对数据的影响。

import torch

def quantize_and_dequantize(tensor, bits):
    torch.manual_seed(42)
    min_val, max_val = tensor.min(), tensor.max()
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = torch.round((tensor - min_val) * scale).int()
    
    # 反量化
    dequantized = quantized.float() / scale + min_val
    return dequantized

# # # 示例
# d = 128  # embedding维度
# l = 8    # 比特数
#
# # 生成一个随机的浮点型embedding
# embedding = torch.randn((5, 10))
#
# # 量化并反量化
# processed_embedding = quantize_and_dequantize(embedding, l)
#
# # 查看原始和处理后的embedding
# print("Original embedding:", embedding)
# print("Processed embedding:", processed_embedding)
# print(quantize_and_dequantize(embedding[2], l))




