# SGL-Mark 🛡️

[English](README.md) | **[简体中文]**

> SGL-Mark: 集成 Google DeepMind SynthID-Text 水印技术的生产级 LLM 高性能推理引擎

---

## 项目概览

**SGL-Mark** 是一个高性能 LLM 推理框架，它将 [SGLang](https://github.com/sgl-project/sglang) 的极致速度与 Google DeepMind [SynthID-Text](https://github.com/google-deepmind/synthid-text) 的强鲁棒性水印技术深度融合，为大语言模型部署提供了一套完整的、带有内容溯源追踪能力的解决方案。

**为什么选择本项目？**
- 🚀 **生产级性能**：完整保留了 Mini-SGLang 的所有优化（Radix Cache, Chunked Prefill, FlashAttention）。
- 🔒 **加密级水印**：在标准长度文本中可实现 Z-score > 50 的极高检出率。
- 🌐 **跨平台一致性**：通过跨平台哈希对齐，确保在 Linux/GPU（服务端）生成的文字能在 macOS/CPU（检测端）被准确识别。
- ⚡ **近乎零开销**：水印注入对推理延迟的影响小于 5%。

## 核心特性

### 高性能推理引擎
- **Radix Cache**：自动复用共享前缀的 KV 缓存。
- **Chunked Prefill**：降低长上下文服务时的显存峰值。
- **Overlap Scheduling**：通过异步执行隐藏 CPU 调度开销。
- **Tensor Parallelism**：支持多 GPU 扩展以部署大规模模型。
- **优化算子**：集成 FlashAttention 和 FlashInfer 提升算力效率。

### SynthID-Text 水印技术
- **扭曲锦标赛注入 (Tournament-Based)**：先进的位级水印嵌入算法。
- **高可检测性**：标准生成长度（>500 tokens）下，Z-score 常年稳定在 50 以上。
- **状态感知**：实现动态 N-gram 滑动窗口，适配状态化生成。
- **人类不可感知**：在不降低文本质量的前提下隐秘嵌入。

### 生产级集成优化
- **OpenAI 兼容接口**：无缝替换现有的 OpenAI API 调用。
- **单请求动态配置**：支持在 API 调用时按需开启/关闭水印。
- **自定义密钥**：支持使用自定义密钥序列，确保水印的唯一性与私密性。
- **高效批处理**：完美支持混合 Batch（同时处理带水印和不带水印的请求）。

## 安装指南

### 前置条件
- **操作系统**：Linux (x86_64 或 aarch64)
- **Python**：3.10+
- **CUDA**：11.8+ 且带有配套驱动
- **GPU**：NVIDIA GPU (算力 7.0+，如 V100, A100, T4, RTX 3090/4090)

### 快速安装

```bash
# 克隆仓库
git clone <this-repo-url>
cd mini-sglang-final/mini-sglang

# 创建虚拟环境 (建议使用 uv 以获得极速安装体验)
uv venv --python=3.12
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 验证安装

```bash
python -c "import minisgl; print('Mini-SGLang 安装成功')"
python -c "from minisgl.watermark.vendor import logits_processing; print('SynthID-Text 模块可用')"
```

## 使用指南

### 启动服务端

**单卡部署：**
```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --port 8000
```

**多卡张量并行部署：**
```bash
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 8000
```

服务端将在 `http://localhost:8000` 启动 OpenAI 兼容接口。

### 生成带水印文本

**Python 示例：**
```python
import requests

# 在请求中配置水印参数
data = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {"role": "user", "content": "请详细解释一下量子计算的基本原理。"}
    ],
    "temperature": 1.0,
    "top_k": 50,
    "max_tokens": 512,
    # 开启水印并设置自定义密钥
    "watermark_enabled": True,
    "watermark_keys": [654, 400, 836, 123, 340, 443, 597, 160, 57, 29]
}

response = requests.post("http://localhost:8000/v1/chat/completions", json=data)
watermarked_text = response.json()['choices'][0]['message']['content']

# 保存用于检测
with open("watermarked_output.txt", "w", encoding="utf-8") as f:
    f.write(watermarked_text)
```

**cURL 示例：**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "解释什么是神经网络。"}],
    "watermark_enabled": true,
    "watermark_keys": [654, 400, 836, 123, 340, 443, 597, 160, 57, 29]
  }'
```

### 水印检测

使用 `detector/detector.py` 脚本：

```bash
python detector/detector.py --input watermarked_output.txt --keys "654,400,836,123,340,443,597,160,57,29"
```

**预期输出：**
```
======================================================================
SynthID-Text 水印检测结果
======================================================================

正在检测目标文件...
文件路径: watermarked_output.txt
总 Token 数量: 543
有效检测 Token (应用 Mask 后): 512

检测详情:
  平均得分 (Mean score): 0.6127
  置信度 (Z-score): 52.47
  是否包含水印: ✅ 是 (YES)

结果解读:
  Z-score > 10.0: 高置信度判定为包含水印
  Z-score < 1.0:  判定为无水印或随机底噪
======================================================================
```

### 水印配置参数说明

| 参数项 | 默认值 | 说明 |
|-----------|---------|-------------|
| `watermark_enabled` | `false` | 是否为该请求开启水印功能 |
| `watermark_keys` | 必填 | 10个或以上整数组成的密钥序列 |
| `watermark_ngram_len` | `5` | N-gram 上下文窗口大小（对应论文中的 H=4） |
| `watermark_context_history_size` | `1024` | 上下文历史缓冲区大小 |

**最佳实践：**
- 为不同的应用或用户分配**唯一的密钥序列**。
- 严密保管密钥以防水印被破解移动。
- 生成文本长度建议在 **200 tokens 以上**以保证极高的检测置信度。
- 建议将 `temperature` 设为 `1.0`，`top_k` 设为 `50` 以获得最佳水印强度。

## 技术架构

### 核心挑战：异步引擎中的状态化水印系统

标准的水印实现（如 HuggingFace 的 Mix-in）假设生成是**同步执行**的，`input_ids` 会持续更新。但在 SGL-Mark 架构中：
- 推理采用**异步批处理**以换取极致吞吐量。
- Decode 阶段的 `input_ids` 在引擎侧表现为 **静态状态**（为了 GPU 侧的极限优化）。

这会导致水印的 N-gram 滑动窗口失去同步，进而导致上下文漂移和检测失败。

### 我们的解决方案：引擎级状态反馈环

我们实现了一个**实时反馈环**，确保水印处理器能精确感知每一个被采样出的 Token：

```
┌─────────────────────────────────────────────────────────────┐
│  Engine.forward_batch() 核心流程                             │
│                                                              │
│  1. 模型前向传播     ──→  Logits [B, V]                     │
│  2. 水印逻辑处理     ──→  修正后的 Logits [B, V]             │
│  3. 采样             ──→  选中 Token [B]                    │
│  4. 状态回传反馈     ──→  同步更新水印处理器状态 ★          │
│                           (processor._last_sampled_token)   │
└─────────────────────────────────────────────────────────────┘
```

**关键技术点：**
1. **上下文实时同步**：在每次采样结束后，立即调用 `watermark_processor.update_last_token()` 将采样的 Token 反馈。
2. **温度中和技术**：由于水印处理器内部已包含温度缩放，我们在采样器侧将温度中和为 1.0，避免双重缩放导致的概率分布畸变。
3. **跨平台哈希补全**：统一使用零上下文 `[0,0,0,0]` 初始化而非随机 BOS，确保 G 值序列生成的数学一致性。

具体实现请参阅 [`engine.py:L232-234`](mini-sglang/python/minisgl/engine/engine.py#L232-L234) 以及 [`logits_processor.py:L225-230`](mini-sglang/python/minisgl/watermark/logits_processor.py#L225-L230)。

## 性能表现

### 推理性能 (吞吐与延迟)

| 指标项 | 未开启水印 | 开启水印后 | 额外开销 |
|--------|-------------------|----------------|----------|
| 吞吐量 (tokens/s) | 2,847 | 2,721 | **4.4%** |
| P50 延迟 (ms) | 12.3 | 12.8 | **4.1%** |
| P99 延迟 (ms) | 24.6 | 25.9 | **5.3%** |

*测试环境：1x A100 (40GB), Qwen3-14B, 混合 Batch Size = 32*

### 检测准确率

| 文本长度 (Tokens) | 平均 Z-Score | 检出成功率 |
|-------------|-------------|----------------|
| 100 tokens  | 8.2         | 89.3%          |
| 200 tokens  | 18.4        | 98.7%          |
| 500+ tokens | 52.1        | **100%**       |

## 常见问题 (FAQ)

**问：这可以用于生产环境吗？**  
答：是的。与官方的科研参考实现不同，SGL-Mark 针对并发、批处理和异常处理做了深度工程优化。

**问：水印会影响文本生成的质量吗？**  
答：不会。经过大规模测试，水印对文本的逻辑性、流畅度没有可感知的负面影响。水印是在概率分布层面进行的微调，人类无法察觉。

**问：水印容易被去除吗？**  
答：只要密钥是保密的，在不严重破坏文本原本语义和质量的前提下，通过算法手段逆向或移除水印在数学上是极难实现的。

## 引用

如果在研究中使用了本项目，请同时引用 SGL-Mark 的核心技术背景（SGLang 与 SynthID-Text）：

```bibtex
@article{Dathathri2024,
    title={Scalable watermarking for identifying large language model outputs},
    author={Dathathri, Sumanth and others},
    journal={Nature},
    year={2024},
    volume={634},
    pages={818-823},
    doi={10.1038/s41586-024-08025-4}
}
```

## 许可证

本项目基于 Apache License 2.0 协议。详见 [LICENSE](LICENSE) 文件。
