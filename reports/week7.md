# Week 7 Report — Memory & RAG

**Dates:** 2026-08-24 to 2026-08-30

Goal (PLAN.md): know when retrieval helps an agent and when it hurts.

## Day 1 — Embeddings: what they are, cosine similarity, embedding models

### Learning

#### 一、Embedding 是什么

Embedding 是把现实世界中复杂、非结构化的数据（文本、图像、音频等）映射为**低维稠密连续向量空间**中的数值向量，使"语义相近的对象"在空间中距离更近。它是 RAG 和 Agent Memory 的底层核心：把"文本/知识"变成向量，再通过向量相似度找到语义相关的内容。

- 语义相近 → 距离近："国王"与"王后"的向量比"国王"与"苹果"更近
- 向量运算蕴含语义关系：king − man + woman ≈ queen
- 不同模态可共享同一空间：文本、图像、音频可以在同一向量空间中对齐

关键概念：

| 概念 | 说明 |
| --- | --- |
| 稠密向量 | 现代 embedding 几乎所有维度都有非零值，信息密度高；对比稀疏向量（one-hot，大部分维度为零） |
| 向量空间 | 所有 embedding 所在的高维数学空间。**分布式表示**：不要指望某一维度代表某一语义，一个语义概念由大量维度共同表达 |
| 维度 | 向量的长度。维度越高理论承载语义越多，常见 384 / 768 / 1024 / 1536 / 4096 |

特性：

| 特性 | 说明 |
| --- | --- |
| 稠密性 | 相比 one-hot 稀疏编码，信息密度高 |
| 连续性 | 向量空间连续，支持插值与梯度优化 |
| 语义保持 | 原始对象的语义关系在向量空间中几何化 |
| 降维 | 把指数增长的离散空间压缩到可控维度 |

演进：

- **静态词嵌入**（Word2Vec、GloVe）：一词一固定向量，捕捉共现关系；无法处理一词多义（"bank" 的"银行"和"河岸"是同一向量）
- **上下文感知嵌入**（BERT、BGE-M3）：同一词在不同上下文中产生不同向量，极大提升语义理解准确性
- **混合嵌入**（BGE-M3）：稠密 + 稀疏结合，既擅长语义理解又精确匹配关键词

典型应用场景：RAG、多模态对齐、偏好推荐。

#### 二、Cosine similarity（余弦相似度）

衡量两个向量**方向**的相似程度，取它们夹角的余弦：

cos(A, B) = (A·B) / (|A|·|B|)，范围 [−1, 1]：1 = 方向完全相同；0 = 正交（无相关性）；−1 = 方向完全相反。

只关心方向、不关心模长：

- 内容相同但长度不同的文档，余弦相似度可能很高 —— "我喜欢猫" vs "我真的非常喜欢猫"应当被认为相似
- 工程实现：先把所有向量做 **L2 归一化**（v̂ = v/‖v‖），余弦相似度就退化为**点积**——预计算一次归一化，大规模检索时只剩矩阵乘，这是 FAISS/Milvus 等向量库提速的基础

与其他度量的对比：

| 度量 | 关注点 | 对模长 | 适用场景 |
| --- | --- | --- | --- |
| Cosine | 方向一致性 | 不敏感（归一化后） | 语义相似度、方向性特征 |
| Dot Product | 未归一化的余弦 | 敏感，保留模长信息 | 推荐系统 |
| Euclidean | 绝对距离 | 敏感 | 绝对位置、聚类密度；A=[1,1] B=[2,2] 方向一致但欧氏距离 √2 ≠ 0 |
| Manhattan | Σ｜a−b｜（各维度绝对差之和） | 敏感 | 对异常值更鲁棒 |
| Angular | arccos(cosine) | 不敏感 | 把相似度转换为角度距离 |

#### 三、Embedding Models：从 Word2Vec 到 LLM 时代

Embedding Model 的任务：Text → Tokenizer → Transformer/Encoder → Pooling → Projection → Semantic Representation（向量）。

三代架构：

1. **静态嵌入** Word2Vec (CBOW/Skip-gram)、GloVe、FastText —— 一词一向量，基于共现统计或上下文预测训练，无法一词多义
2. **上下文嵌入** BERT、RoBERTa、DeBERTa —— Transformer Encoder 自注意力捕获上下文；注意原始 BERT 的 [CLS] token 不适合直接做句子表示（语义漂移问题）
3. **句子/文档嵌入** Sentence-BERT、E5、BGE、GTE —— 专门优化句子/段落级表示，核心训练技术是**对比学习 (Contrastive Learning)**

**对比学习**——现代 embedding 模型的核心训练范式：让语义相似的样本靠近、不相似的远离。给定锚点 x：正样本 x⁺（与 x 语义相同，如同句子的不同增强版本）、负样本 x⁻（语义不同）。InfoNCE 损失：

L = −log( e^{sim(x,x⁺)/τ} / ( e^{sim(x,x⁺)/τ} + Σ_{x⁻} e^{sim(x,x⁻)/τ} ) )

τ 是温度参数，sim 通常是余弦相似度——让正确答案的相似度在候选集合中尽可能高。这就是 embedding 模型最终能用于 retrieval 的原因。

**对称 vs 非对称 embedding**（工程选型要"注意看"的点）：

- **对称**：query 和 document 用同一个模型、同一写法
- **非对称**：query 和 document 用不同模型或不同 prompt 前缀（如 E5/BGE 系列：query 加 "query:" 前缀、passage 加 "passage:" 前缀）；检索任务通常选非对称设计

模型选型考量：上下文窗口（长文档如论文需要 8192 tokens 级别）、领域表现、向量维度（MRL 技术支持训练后灵活降维省成本）、语言支持（中文能力）、部署与成本（商业 API 便利 vs 开源本地部署的控制权）、检索准确率。

| 维度 | 建议 |
| --- | --- |
| 模型选择 | 优先 MTEB leaderboard 前列的开源模型（BGE、GTE 系列） |
| 向量维度 | 384–768 维通常足够，效果与存储间权衡；可用 PCA 降维 |
| 量化 | 生产环境可用 int8 或 binary 量化减少 4–32× 存储，轻微损失精度 |
| 索引结构 | 百万级用 FAISS IVF；十亿级用 HNSW、ScaNN 或 Milvus/Pinecone |
| 重排序 | 先 embedding 召回 Top-K，再用 Cross-Encoder 精排 |

#### 四、Memory 与 RAG：同一套检索机制，两种用途

Memory = 通过语义检索找到过去相关经历。RAG = 给 Agent 一个外部长期知识库；Memory = 给 Agent 一个关于过去交互的长期记忆：

| | RAG | Memory |
| --- | --- | --- |
| 存什么 | 外部知识 | 历史信息 |
| Embedding | 文档 chunk | 对话/事件/事实 |
| Query | 用户问题 | 当前上下文 |
| Retrieval | 找知识 | 找记忆 |
| Vector DB | 知识库 | Memory Store |
| LLM | 生成答案 | 使用记忆 |

Memory 不只是"把历史知识存为向量"，完整的 agent memory 包括：Short-term Memory（当前 conversation context）、Long-term Memory（Semantic「我知道什么」/ Episodic「过去发生过什么」/ Procedural）、Retrieval（Vector Search + Keyword Search + Reranking）。Embedding 主要作用于 Semantic Memory 和 Retrieval。

**纯语义检索的盲区**："2025年3月12日，A公司的订单 A-123456 发生了什么？"——精确日期、精确数字、订单号、人名、产品型号、代码，纯 semantic similarity 都不擅长。所以现代 RAG 常用**混合检索**：Vector Search + BM25/Keyword Search → 候选合并 → Reranker 精排 → Top-K。

两阶段分工：Embedding 负责**召回**（query → embedding → 快速找到 Top 50）；Reranker 负责**精排**（query + document 一起输入 → 重算相关性 → Top 5）。

完整 RAG 流水线（Memory 只是把 Raw Documents 换成 Past Conversations / Past Events / User Facts / Agent Experiences）：

```
  Raw Documents ──► Chunking ──► Embedding Model ──► Vector Embeddings
                                                        │
                                                   Vector Index / Vector DB
                                                        │
User Query ──► Embedding ──────────────────────────────┘
                                                        │
                                                Similarity Search
                                                        │
                                                 Top-K Candidates
                                                        │
                                                    Reranker
                                                        │
                                                Relevant Context ──► LLM ──► Answer
```

RAG 的效果与每个环节都有关：chunk 大小、query 和 document 的写法差异、top-k 太小、embedding 模型的中文能力……（Day 2 的 chunking 就是第一个可变环节。）

### Experiment — 50 篇短文本的 embedding + 纯 numpy top-k 检索（无向量数据库）

#### Question

余弦 top-k 检索在"50 篇同一领域短文本"语料上能做到多准？纯 numpy 的三行实现与向量数据库之间的差距到底是什么？

#### Setup

- **Model:** DashScope `text-embedding-v3`，1024 维（`text-embedding-v4` 与 maas 端点也验证可用）
- **Tasks:** 50 篇短文本 = 10 主题 × 5 篇，全部摘自已有的 week 1–6 笔记（agent_loop / tools / context / evals / temperature / memory_rag / cost_tokens / mcp / multiagent / planning），主题标签即 ground truth；12 个 query = 10 个主题查询 + 1 个陷阱查询（"何时该用向量数据库"）+ 1 个中文查询（跨语言演示），k=5
- **Varied:** 无——构建日，单一条件（无 A/B）
- **Held constant:** 语料、模型、维度、k；评分对给定 embedding 矩阵是确定性的
- **Runs:** 1 次（7 批 embedding 调用，共 1,487 tokens，约 3 秒）；离线 selftest 10/10 通过后才调用 API
- **Metrics:** hit@k（gold 5 篇中进了 top-k 几篇）、MRR（首个 gold 命中的 1/rank）、precision@k；外加两个"聚团"指标：主题内 vs 主题间平均余弦、最近邻同主题比例

**检索实现（本日核心，3 行 numpy）：**

```python
E_hat = E / np.linalg.norm(E, axis=1, keepdims=True)  # L2 归一化
sims  = E_hat @ q_hat                                  # 全部余弦相似度，一次矩阵乘
idx   = np.argsort(-sims)[:k]                          # 降序取 top-k
```

余弦相似度 = 归一化向量的点积；对 n=50 来说，O(n·d) 的一次 matmul 就是全部检索。向量数据库只是这同一操作的**索引层**——n 大到百万级、需要 ANN 近似索引、持久化、分片时才值得引入。这正是"不用向量库，数组就够了"的用意。

#### Result

| Query | gold 主题 | hit@5 | MRR |
| --- | --- | ---: | ---: |
| q_agent_loop | agent_loop | 3/5 | 1.000 |
| q_tools | tools | 2/5 | 1.000 |
| q_context | context | 3/5 | 1.000 |
| q_evals | evals | 2/5 | 1.000 |
| q_temperature | temperature | **5/5** | 1.000 |
| q_memory_rag | memory_rag | **5/5** | 1.000 |
| q_cost_tokens | cost_tokens | 1/5 | 1.000 |
| q_mcp | mcp | 4/5 | 1.000 |
| q_multiagent | multiagent | 3/5 | 1.000 |
| q_planning | planning | 1/5 | 1.000 |
| q_vectordb（陷阱） | memory_rag | 2/5 | 1.000 |
| q_evals_zh（中文） | evals | 2/5 | 1.000 |

| 全局指标 | 值 |
| --- | ---: |
| mean MRR | **1.000**（12/12 首命中即正确主题） |
| mean hit@5 | 2.75 / 5 |
| mean P@5 | 0.55 |
| 主题内 vs 主题间平均余弦 | 0.5286 vs 0.4700（差 +0.059） |
| 最近邻同主题比例 | 0.52 |

- **Top-1 完全可靠，Top-5 被相邻主题污染。** 12 个 query 的首个命中全部是正确主题；但 hit@5 只有 2.75——q_planning 的 #2–3 被 multiagent 文本占据，q_cost_tokens 被 context/mcp 填满。
- **主题内/主题间余弦只差 +0.059**，主题间平均余弦高达 0.47：50 篇文本全部关于 AI agent，语料同质性决定了邻域天然模糊。
- **跨语言检索零成本生效**：中文 query（如何衡量评估任务的通过率？）直接命中英文 evals 文本，无需任何特殊处理。
- **离线 selftest 是理解力的证明**：10 项检查（numpy 数学对照朴素循环、合成近重复语料 hit@5=1.0、余弦 ≡ 单位向量点积恒等式）先于任何 API 调用通过，并抓住了实现中的 2 个 bug（关键字参数语法错误、query 采样错主题）。

#### Surprise

1. **MRR=1.0 与 hit@5=2.75 同时成立**——检索空间"有结构"和"邻域干净"是两回事。所有 query 都落进了正确的邻域，但邻域互相渗漏（planning ↔ multiagent、cost ↔ context ↔ mcp）。这不是 text-embedding-v3 的失败，而是"语料全是一个领域"的必然——也正是 Day 2 说 **chunking 策略主导 RAG 质量**的原因：召回邻域模糊时，塞进 prompt 的东西决定有用性。
2. **陷阱 query 被"内容敏感"地接住了**：q_vectordb 的首命中不是 memory_rag 主题里最泛的"embedding 是什么"文本，而是"50 个向量不需要向量库"那条——检索命中的是**语义上最对题的具体文本**，而不仅是主题归属。
3. **学习笔记里的盲区在自家语料上重现**：笔记写的"精确数字/专名难检索"正是 q_cost_tokens（文本满是 565k、37:1、1,909 等数字）表现最差（1/5）的候选解释——数字在 embedding 空间里区分度低，充满数字的 cost/context/mcp 文本互相占据邻域。

#### Next question

- Day 2：chunk 大小与 overlap 如何主导 RAG 质量（同一批语料变成"文档 → chunk → 检索"）。
- Day 3：检索作为 agent 的**工具**（按需调用）vs 全量塞进 context，5 个问题上的对比——何时检索帮助 agent、何时伤害它。
- 混合检索（vector + BM25）对 q_cost_tokens 这类"精确值"查询的提升，是 Day 4+ memory 检索的候选实验。

---

## Day 2 — Chunking: sizes, overlap, why chunking strategy dominates RAG quality

### Learning

**核心观点**：Embedding 决定"文本如何被理解"，Chunking 决定"文本如何被切成可检索的知识单元"。垃圾进垃圾出，在 RAG 里具体表现为"错误的分块 → 必然的错误生成"。

**为什么必须分块**：极长的文档（如产品说明书：第一章功能概述、第二章使用说明、第三章故障排查、第四章保修条款……）如果整篇只做一个 embedding，这个向量需要同时"代表"所有主题，无法准确表达任何一个具体语义。

| 约束 | 说明 |
| --- | --- |
| **上下文长度限制** | Embedding 模型和 LLM 都有最大输入 token 限制（BERT 时代 512、现代常见 8192–128K），超长文档无法一次性编码 |
| **注意力稀释** | Transformer 的注意力在长序列中效果衰减，关键信息被淹没（week 3 的原始 HTML 灾难同理） |
| **检索精度** | 过长的 chunk 语义混杂，查询与 chunk 的匹配精度下降 |
| **计算与存储成本** | 向量维度固定时，chunk 数量直接决定存储量和检索计算量 |

**Chunking 的本质**不是"把文本切短"，而是把长文档转换成适合检索的**知识单元**——每个 chunk 应尽可能自洽、可检索、能直接支撑回答。好 chunk 的四条标准：

1. **Semantic Coherence** —— 内部内容属于同一个主题
2. **Sufficient Context** —— 包含理解该主题所需的上下文
3. **Retrieval Specificity** —— 不同时混入太多无关主题
4. **Answerability** —— 最好能直接支撑一个问题的答案

#### 一、分块大小（Chunk Sizes）：粒度与语义的博弈

大小可按 tokens / characters / words / sentences / paragraphs 定义：

| Chunk Size | 优点 | 缺点 |
| --- | --- | --- |
| 很小 | 精确、噪声少 | 上下文不完整 |
| 中等 | 平衡 | 需要调参 |
| 很大 | 上下文完整 | 语义混杂、召回不精确 |
| 极大 | 几乎不丢上下文 | 检索失去意义、成本高 |

- **小 chunk（128–256 tokens）**：检索精度高、噪声少、embedding 聚焦；但信息分散、召回率可能低、LLM 需拼接多个 chunk、向量库压力大（数量多）
- **大 chunk（1024–2048+ tokens）**：语义完整性高、保留更多上下文；但可能超出上下文窗口、embedding 被"平均化"而模糊

| 场景 | 推荐 size | 理由 |
| --- | --- | --- |
| **代码检索** | 50–150 tokens | 函数/类级别，逻辑单元紧凑 |
| **FAQ / 客服知识库** | 100–200 tokens | 问答对简短，需精准匹配 |
| **通用文档 RAG** | 256–512 tokens | 平衡精度与上下文，最常用 |
| **学术论文 / 法律文档** | 512–1024 tokens | 需保留论证链条和引用关系 |
| **长叙事文本（书籍、报告）** | 1024–2048 tokens | 论点跨度大，需 broader context |
| **多模态（图文交错）** | 视内容单元而定 | 按段落、幻灯片页、章节边界切割 |

> 单位注：上表是 **tokens**。工程上按字符分块更便宜但更糙；本周实验用字符分块，实测 400 chars ≈ 平均 165 embedding tokens。

**动态/语义分块（Semantic Chunking）**：不再硬性规定 token 数，而是用 NLP 工具（句子分割器）或模型计算相邻句子的 embedding 变化率，当语义出现陡峭转折（换主题）时自动切分，保留完整的段落语义边界。

#### 二、分块重叠（Overlap）：对抗"边界截断"的免疫机制

相邻两个分块互相包含的字符/token 数量，通常设为 chunk 大小的 **10%–20%**。

- **为什么需要**：无论块大小如何设定，硬性切割（如按第 512 个字符一刀切）必然切断句子的语法结构或实体关系
- **核心机制**：滑动窗口（Sliding Window）让下一个 chunk"继承"上一个 chunk 的尾部内容，避免重要语义恰好被切断
- **"双保险"作用**：① 上下文桥接——指代词（如"上述算法"）在切割后仍能在新 chunk 开头找到先行词；② 排序纠正——关键实体被截断在两个 chunk 边缘时，重叠确保它在两个 chunk 中都以完整形态出现，即使一个 chunk 因向量偏移未被召回，另一个仍能兜底
- **不是越大越好**：两个 chunk 大量重复，带来存储/计算消耗、检索候选增多、重复上下文浪费 LLM token。本周实测：800 字符 chunk 加 100 字符 overlap（12.5%，恰好落在 10%–20% 经验带内）后 chunk 数 466→533、embedding tokens **+14%**，但答案质量 **2/6→5/6**——这笔开销花得值（见 Experiment）

**Chunk Efficiency**（工程指标）：size=500、overlap=100 时，每个新 chunk 实际只新增 400 token，重复率 20%。

**智能 Overlap 策略**：句子边界对齐（overlap 区域在句子边界截断，而非粗暴按 token 数）；语义滑窗（步长 < chunk size）；层级重叠（父 chunk 与子 chunk 并存，检索时联合使用）。

#### 三、为什么分块策略"绝对主导"RAG 质量

反直觉但重要的共识：在典型 RAG 系统中，chunking strategy 对最终生成质量的影响，**往往大于 embedding 模型的选择或 prompt engineering**。原因是它在 RAG 四步流程（Ingest → Retrieve → Rerank → Generate）中处于"基石漏斗"位置：

1. **Chunking 决定信息密度，是不可逆瓶颈**——答案是否存在于候选集，在分块阶段就已决定。信息切坏了，embedding 模型再强也无法恢复被拆开的语义关系。
2. **分块直接干扰 embedding 的"高频特征"捕获**——块过大时自注意力因序列过长而分散（Attention Dilution），向量无法聚焦核心主题（Topic Drift）。本周 fixed_800 vs fixed_400 的 2/6 vs 5/6 就是实测证据（见 Experiment）。
3. **分块钳制 LLM 的上下文窗口利用率**——上下文预算 = k × chunk size，chunk 大小和 k 必须一起选；良好分块（小且准）能用极少的 Top-K 填满窗口，达到最高信噪比。
4. **决定"细粒度事实"与"宏观逻辑"的平衡**——金融研报/法律合同既有具体数值又有结论概要，单一块大小无法兼顾。工业级做法是多路分块（Multi-path Chunking）：同时生成大块（给 LLM 通读）和小块（给向量检索），最后合并去重。

不同 embedding 模型对 chunk 长度的敏感度不同：

| 模型类型 | 对 Chunk Size 的敏感性 |
| --- | --- |
| **BERT-based（512 max）** | 严格受限，超长直接截断，chunking 必须精准 |
| **Long-context Embedding（8K+）** | 可接受较大 chunks，但仍存在注意力稀释 |
| **Late Interaction（ColBERT）** | 对 chunk 大小较鲁棒，token-level 交互补偿部分缺陷 |
| **Multi-vector（BGE-M3）** | 可结合稀疏表示，缓解 dense embedding 的粒度问题 |

#### 四、合理的做法

- **优先利用文档本身的结构**（章节、段落、句子），不要固定 token 一刀切
- **递归字符分割（Recursive Character Text Splitting，LangChain 标准做法）**：按优先级分层切割——优先级 1: 段落边界 (\n\n)、优先级 2: 换行 (\n)、优先级 3: 句子 (. ! ?)、优先级 4: 单词 (空格)、优先级 5: 字符；高优先级边界切不动才降级
- **语义分割**：计算相邻片段的 embedding 相似度，语义明显变化处切分——但注意**语义变化 ≠ 知识边界**
- **命题分块（Propositional Chunking）**：把文档分解为原子事实（propositions），每个 proposition 是自包含的语义单元
- **Metadata 是"隐形武器"**：chunk 文本模糊时靠 metadata 补全——document=产品使用手册、chapter=故障排查、section=网络配置、page=42、product=路由器A3。实际 RAG = **Chunk + Metadata**
- **Parent-Child Chunking**：小 chunk 用于检索（precision ↑），命中后返回其父 chunk 给 LLM（context completeness ↑）——Query → Child Embedding → 找到 Child 3 → 找到其 Parent → 把 Parent Context 给 LLM
- **Multi-Vector Retrieval**：同一文档生成 summary / paragraph / question 多种 embedding，多路检索后融合结果
- **不同数据不同策略**：表格不能把表头和内容分开、论文按章节组织、代码按类和函数定义

现代 RAG 的思路不是"把所有东西都塞给 LLM"，而是：**尽可能少 + 尽可能相关 + 尽可能完整**的 context——在最小噪声下，给信息密度最高的上下文。

#### 五、如何判断 chunk 好不好——用 evaluation，别凭感觉

检索层：Recall@K、Precision@K、MRR、nDCG；生成层：Answer Accuracy、Faithfulness、Citation accuracy；系统层：Latency、Token Cost。

| 指标 | 测量方法 |
| --- | --- |
| **检索准确率 (Retrieval Accuracy)** | Top-K chunks 是否包含回答查询所需的信息 |
| **上下文充分性 (Context Sufficiency)** | 单个 chunk 是否自包含，无需外部信息即可理解 |
| **信息冗余度 (Redundancy)** | 相邻 chunks 的 overlap 是否造成检索结果重复 |
| **边界完整性 (Boundary Integrity)** | 句子/段落是否在边界处完整保留 |
| **端到端答案质量** | 最终 LLM 生成的答案准确率（黄金标准） |

**关键：不要只看最终结果，要拆成检索和生成两层看**——回答错了，是 Chunking/Retrieval 没召回正确内容，还是 LLM 拿到正确内容仍答错？两者的修法完全不同。本周实验用 hit@doc 评检索层、evidence grader 评生成层，正是这个拆法（详见 Experiment 的失败分类）。

**Chunking 设计清单（八问）**：① 数据是什么（PDF/HTML/Word/Code/Table/FAQ）→ ② 文档本身有没有结构（Chapter/Section/Paragraph/Function）→ ③ 自然知识单元是什么（Question/Clause/Paragraph/Function）→ ④ query 通常是什么形式（事实查询/综合查询/精确数字/多跳问题）→ ⑤ 需要多大上下文才能回答（100/500/1000 tokens）→ ⑥ 边界是否容易破坏语义（决定是否设 overlap）→ ⑦ 是否需要 Parent-Child（小 chunk 检索 + 大 chunk context）→ ⑧ 最后通过 evaluation 调参（Recall@K、Precision@K、MRR、nDCG、Answer Accuracy）。

**关键原则总结**：

1. **Chunk 必须自包含 (Self-contained)**——一个 chunk 应包含回答某类查询所需的完整上下文，避免代词悬空、引用丢失
2. **边界尊重语义 (Semantic Boundary Respect)**——优先在段落、章节、逻辑单元边界切割，而非强制固定长度
3. **Overlap 是保险而非解决方案**——缓解边界问题，但不能替代合理的 chunk size 和边界选择
4. **One Size Does Not Fit All**——同一知识库中不同类型内容（表格 vs. 正文 vs. 代码）应使用不同的 chunking 策略
5. **Chunking 与检索策略协同**——小 chunk + 重排序（Reranking）往往优于大 chunk + 朴素检索（本周实验中，小 chunk 不加重排序就已经赢了大 chunk）

RAG Quality ∝ Chunking Strategy × Embedding Model × LLM Capability——其中 Chunking Strategy 是乘数因子：**当分块失败（信息被切坏）时，无论后两者多强，结果都趋于零**。一个优秀的 chunking 策略，是在语义完整性、检索精度、计算效率和存储成本之间找到针对具体场景的帕累托最优。

### Experiment — 迷你 RAG 流水线：chunk → embed → retrieve → stuff（4 种分块策略 × 6 个问题）

#### Question

在真实笔记文档（`reports/week1.md`–`week6.md`，6 篇、370,288 字符）上跑通完整 RAG 流水线：分块策略（大小 / overlap / 结构感知）如何影响端到端答案质量？以及——检索本身到底值不值（对比无检索基线）？

#### Setup

- **Model:** embedding = DashScope `text-embedding-v3`（1024 维，day 1 客户端原样复用）；生成 = `qwen3.7-plus`，temperature 0
- **Tasks:** 6 个问题，每个对应一篇 week1–6 报告；gold 事实是事先从报告里核实的精确数字（4/5 vs 0/5、97.5%+100%、30,000+10%、91%、33.6%、45.6%+37:1）。评分 = evidence presence：gold 字符串全部出现在归一化（去空格/逗号/全角）后的答案中。问题英文、语料中文（day 1 已验证跨语言检索可用）
- **Varied:** chunking 策略（唯一变量）——`fixed_400` / `fixed_800` / `fixed_800_o100` / `para_800`（段落合并，target 800），外加 `no_retrieval` 基线。单一变量对比：大小（400 vs 800）、overlap（800 vs 800+100）、结构感知（para_800 vs fixed_800）
- **Held constant:** 文档、embedding 模型、k=4、chat 模型、温度
- **Runs:** 1 次全量（944 秒，约 69.3 万 tokens）；先跑离线 selftest 23/23（chunker 重构无损、overlap 不变量、grader 归一化、检索集成）再调 API
- **Metrics:** accuracy（gold 事实齐全）、hit@doc（top-k 中来自 gold 文档的比例——把检索和生成拆开看）、prompt tokens/问（stuff 的成本）

#### Result

| Condition | chunks | accuracy | hit@doc | prompt tok/q |
| --- | ---: | ---: | ---: | ---: |
| no_retrieval | — | 0/6（0.000） | — | 68 |
| fixed_400 | 929 | **5/6（0.833）** | **0.708** | 775 |
| fixed_800 | 466 | 2/6（0.333） | 0.625 | 1,340 |
| fixed_800_o100 | 533 | **5/6（0.833）** | 0.667 | 1,253 |
| para_800 | 602 | 3/6（0.500；人工复核 4/6） | 0.625 | 1,178 |

9 个失败（4 策略 × 6 问 − 15 通过）拆成三类，正好对应学习笔记里"拆两层评估"的说法：

1. **检索失败（模型诚实回答 "I don't know from the context"）**——q_week6 四种策略全挂：45.6%/37:1 的事实位于 week6.md 开头，但 query 的措辞与 week5 大量成本段落更匹配，top-4 全被 week5/week3 占据；fixed_800 的 q_week1 只召回 1/4 正确文档的 chunk，另外 3 个错文档 chunk 污染了上下文；800 粒度下的 q_week5 落进了正确文档的错误区域。
2. **生成失败**——fixed_800 的 q_week4 从部分上下文幻觉出"100%（11 out of 11）"，自信且错误。噪声 stuffing 会制造权威感——day 1 语料里的那课："retrieval hurts when it stuffs noise into the context"。
3. **Grader 假阴性**——para_800 的 q_week4 答案"10 out of the 11 … fraction 10/11"完全正确，但 evidence grader 只认"91"。人工复核改判通过（与 week 4 报告同样的裁定流程）。

#### Surprise

1. **"大块上下文更完整"的直觉输了。** 400 vs 800 的单一变量对比是 **5/6 vs 2/6**。大 chunk 把多个主题混进一个 embedding（Topic Drift 的实测版），召回排序变差（hit@doc 0.625 vs 0.708）；而且小块每问更便宜（775 vs 1,340 prompt tokens）——上下文预算 = k × chunk size，小块用同样的 k 塞进更少但更纯的内容。
2. **Overlap 是廉价且有效的保险。** 800 加 100 字符 overlap（12.5%）后 **2/6 → 5/6**，代价只有 +14% embedding tokens。每个事实在 ~2 个 chunk 中以完整形态出现，边界位置不再决定召回成败——与学习笔记里"双保险"机制的描述一一对应。
3. **无检索基线 0/6，但参数知识是模糊且不稳定的。** 5 个私有事实模型直接拒绝（"I do not know"）；唯一有参数知识的 q_week3（原始 HTML 的 token 规则）在 smoke run 答出"15,000–30,000 tokens"并通过、全量 run 答成"10,000–50,000"而失败——同一问题、同一温度 0，推理模型的参数记忆给的是**范围而非精确值，且 run 间漂移**。检索补的正是"精确"。
4. **hit@doc 与 accuracy 会背离。** para_800 与 fixed_800 的 hit@doc 相同（0.625），但答案质量更好（复核后 4/6 vs 2/6）：召回对了不代表 chunk 内容能回答问题——不拆两层评估就会把生成失败误诊为检索失败。para_800 在相同粒度上赢过 naive fixed_800，但仍输给 fixed_400。
5. **"chunking strategy dominates RAG quality" 从共识变成了自家数据。** 同一语料、同一模型、同一 k=4，只换分块，accuracy 从 2/6 跨到 5/6。

#### Next question

- Day 3：把这条流水线包装成 `search_knowledge` 工具（agent 按需调用）vs 全量塞 context，5 个问题上对比——何时检索帮助 agent、何时伤害它。
- 把 evidence grader 换成 week 4 的 LLM judge（本次已现假阴性）；day 6 用 ≥5 runs 重测 400 vs 800 的方向（本次单次运行，方向可信、方差未测）。
- 混合检索（vector + BM25）对 q_week6 这类"精确数字 + query 措辞错位"的查询是否有效，是候选实验。

---

## Day 3 — RAG for agents: retrieval as a tool vs always-stuffed context

### Learning

#### 一、两种哲学：谁来决定 agent "看什么"

把 RAG 装进 agent 时，有两个根本不同的答案：把检索做成 agent 可以调用的**工具**（Retrieval as a Tool），还是在每次推理前由**系统**把资料塞进上下文（Always-Stuffed Context）。核心矛盾是 **agent 的自主性 vs 系统的确定性**：工具化检索把信息获取的主动权交给 agent，代价是引入决策风险；全量填充消除检索决策的不确定性，代价是受限于上下文窗口并可能信息过载。

| 维度 | Retrieval as a Tool（按需检索） | Always-Stuffed Context（全量预填充） |
| --- | --- | --- |
| 决策者 | Agent 自主决定何时检索、检索什么 | 系统在推理前静态注入 |
| 上下文形态 | 动态、按需、增量加载 | 静态、全量、始终可见 |
| 信息边界 | Agent 只能看到它主动请求的内容 | Agent（理论上）拥有全部相关知识 |
| 成本模型 | 可变：随"查了几次"增长，与语料规模无关 | 固定：随"语料多大"增长，每次推理全付 |
| 延迟特征 | 多轮、可能累积 | 单轮、确定性 |
| 反馈机制 | Agent 根据观察调整检索策略 | 无反馈，一次性的上下文供给 |
| 失败模式 | 不检索（幻觉）、错误检索、无限检索 | 上下文遗漏、注意力稀释 |
| 可扩展性 | 知识库几乎无上限 | 受限于窗口与成本 |

粗判据：简单文档问答、静态知识库客服，全量填充性价比最高、维护简单；复杂研究、多源整合、金融法律分析，必须走检索即工具——牺牲一点延迟，换来系统级容错与推理透明度。

#### 二、Retrieval as a Tool：检索进入决策循环

检索不再是固定的 pipeline，而是工具列表里的一个函数——与 ReAct 一脉相承：agent 自己决定要不要查、查什么、查几次、查到之后继续推理还是继续查。触发信号：

| 触发条件 | 示例 |
| --- | --- |
| 不确定性识别 | "我不确定差旅报销政策是否包含国际航班…" |
| 事实缺口 | 回答需要具体数据（销售额、客户名、API 签名） |
| 时效性需求 | "最新版本支持哪些功能？" |
| 领域外问题 | 超出参数化知识范围的内容 |
| 验证需求 | 生成答案前交叉验证 |

一次典型的检索循环（ReAct）：

```text
Thought: 用户问 Q3 净利润，训练数据里没有这个数字，需要查内部财务文档。
Action:  knowledge_retrieval(query="Q3 2025 净利润 net profit")
Observation: [片段1: "Q3 净利润 2.4 亿美元…"] [片段2: "同比增长 15%，云服务驱动…"]
Thought: 信息足够，可以回答。
Final Answer: Q3 2025 净利润 2.4 亿美元，同比增长 15%…
```

一次检索也不必是终点：agent 可以把问题拆成多个子查询做 multi-step RAG（Round 1 查"Transformer 原理"→ 发现需要"自注意力"→ Round 2 查"Self-Attention 数学推导"→ Round 3 查"QKV projection"→ 获得完整理解）。**Agent 生成的 query 质量直接决定 RAG 效果**，常见策略：

| 策略 | 说明 |
| --- | --- |
| 原始查询直用 | 用户问题直接当 query（简单，但常非最优） |
| 查询重写 (Query Rewriting) | 把模糊/多义问题转成精确的检索语句 |
| 假设文档嵌入 (HyDE) | 先让模型生成假设答案，用假设答案的 embedding 检索相似文档 |
| 子查询分解 | 复杂问题拆成多个独立的检索查询 |
| 对话历史感知 | 结合多轮上下文生成 query，处理指代消解 |

检索工具本身也应按领域拆分（`search_docs()` / `search_orders()` / `search_tickets()` / `search_customers()`），而不是一个大而全的 `search`——工具粒度越贴合任务，agent 的选择越不容易错。这是 week 5 工具粒度与描述实验结论的直接迁移：**工具描述就是 when-to-call 的 prompt engineering**。

#### 三、自主权的代价：工具化检索必须配套治理

工具化检索有三个原生失败模式：**该检索时没检索**（直接幻觉）、**错误检索**（query 偏离意图）、**无限检索**（反复查但无法综合，循环陷阱）。所以成熟的 Agentic RAG 需要一整套护栏：

- **Tool Policy**：哪些问题允许/禁止检索，什么场景用什么工具
- **Iteration Limit**：硬性轮次上限（如 MAX_TURNS）——真正的兜底
- **Retrieval Budget**：检索次数预算——注意**提示词里的预算是软约束**：day 3 实验里模型在 q_week4 上超了预算（提示"最多 3 次"、实际 5 次），而且正是超出的检索找到了答案；硬护栏必须机制化
- **Query Constraints**：query 长度/格式约束，防止检索词退化
- **Reranking**：对 embedding 的 top-k 用更强模型或规则重排序——day 1 的教训是语料同质时邻域不干净（intra/inter 余弦只差 0.059），rerank 是第二道滤网
- **Evaluation**：检索触发率、query 精准度、幻觉率（见第五节指标表）

把"是否检索"抽象成决策问题：给定 query 和已有 context，检索的期望收益是否值得成本？`expected_utility = E[信息增益] - 检索成本`。增益低、代价高就不检索；增益高、代价低就检索——p(retrieve | query, context)。

两个容易被忽视的设计点（本次实验实测）：
1. **检索结果每轮都重新进入上下文。** 一次检索返回的 chunks 作为 tool result 写进消息历史，之后每一轮推理都全量重发——检索即上下文增长（week 6 的定律在检索工具上的重现）。因此要控制返回规模（top-k、截断标记），并返回结构化、可解析的结果信封（week 3 的 status/data/truncated），agent 才可能在多轮里正确使用结果。
2. **结果里带来源与分数。** 每个 chunk 标注 doc + similarity，并在系统提示里约束 agent "只依据工具结果回答"——这是对抗检索幻觉（捏造来源、引用不存在的段落）的最低配置。

#### 四、Always-Stuffed Context：检索下沉为基础设施

另一种做法：LLM 每次推理前，系统自动检索/整理资料并注入 System Prompt 或 User Message 前缀。Agent 始终在一个"已被充分告知"的环境里推理——检索从 agent 的决策循环中剥离，下沉为基础设施层的隐式操作。它是固定流程、简单，是传统工作流式 RAG 的默认做法。

```text
system_prompt = f"""
你是企业智能助手。以下是与你工作相关的背景信息：
[知识库摘要]
{retrieved_contexts}   # 预检索的 Top-K 文档
[用户档案]
{user_profile}
[当前任务上下文]
{task_context}
请基于以上信息回答用户问题。如果信息不足，请明确说明。
"""
```

"全量填充"不是只有"把整个知识库塞进去"一种做法，而是分层策略：

| 策略 | 实现方式 | 适用场景 |
| --- | --- | --- |
| 全量静态注入 | 固定知识库（如产品手册）完整放入 system prompt | 知识量小（< 窗口 20%）且稳定 |
| 路由预筛选 | 先做粗粒度分类，注入对应领域的知识子集 | 多领域知识库 |
| 摘要压缩 | 小模型/离线流程把长文档压成结构化摘要 | 知识量大但可压缩 |
| 层级上下文 | 高层概览 + 关键细节的"金字塔"结构 | 宏观理解 + 微观执行 |
| 记忆窗口 | 滑动窗口保留最新/最相关的交互历史 | 长对话会话 |

它的优势是确定性、单轮低延迟、简单心智模型、不会因 agent 误判而漏检；风险是窗口瓶颈、注意力稀释（Lost in the Middle）、成本刚性（无论是否需要都付全额）、信息过时，以及缺乏适应性（无法针对特定问题深入检索，只能依赖预筛选的"运气"）。

#### 五、如何选择：判断标准与融合架构

设计时别急着选边，先回答一组问题：**检索什么时候发生？谁决定检索？query 由谁构造？可以调用几次？检索后是否允许继续推理/再检索？是否有 token 上限？检索是一个工具还是多个工具？** 答案不同，架构就不同。

生产实践的融合思路——**默认提供少量高价值 context + agent 按需检索**：

1. **Smart Router（智能路由层）**：前置意图分类器快速判断问题的知识依赖性——事实性/确定性问题降级为"始终填充"，一步到位；推理/探索性/模糊问题升级为"检索即工具"，激活 agent 循环。纯工具化检索的延迟在高频场景不可接受，路由层按问题类型混合两种范式；在 agent 模式下，检索工具还可以附带置信度评分，低于阈值自动触发"重写-重检"循环（ReAct 的 Act → Observe）。
2. **分层架构**：
   - Layer 1 Always-Stuffed（基础层）：注入角色定义、固定业务规则、高频知识摘要——建立世界观与约束边界；
   - Layer 2 Tool-Based Retrieval（动态层）：向量检索、SQL、API——按需获取细节、实时数据、个性化信息；
   - Layer 3 反思性再检索（Reflexive Retrieval）：对答案置信度低时主动发起验证性检索——自我纠错、减少幻觉。
3. **与记忆系统结合**：短期记忆（当前对话历史）用 Always-Stuffed；长期事实性记忆用向量库的 Tool-Based Retrieval；长期程序性记忆（工作流、技能定义）用 Always-Stuffed。

为什么"检索即工具"是通往通用智能体的必经之路？agent 的本质是与环境交互。全量填充模式下知识库是静态"备忘录"，agent 没有自主权；工具模式下知识库变成动态"外部世界"——既解决信息过载（企业知识库动辄数十亿 token，只有工具模式支持"先看目录再翻章节"），又赋予 agent"好奇心"（生成中遇到生僻概念可以暂停生成、先去查定义，提升答案严谨性）。

无论哪种范式，都要监控评估：

| 指标 | 工具化检索 | 全量填充 |
| --- | --- | --- |
| 检索触发率 | agent 识别信息缺口的准确率 | N/A |
| 检索精准度 | 生成 query 与相关文档的匹配度 | N/A |
| 上下文利用率 | N/A | 注入内容实际被引用的比例 |
| 幻觉率 | 未检索时的错误回答比例 | 上下文遗漏导致的错误比例 |
| 端到端延迟 | 平均工具调用次数 × 检索延迟 | 固定（随上下文长度） |

一个可以直接复述的面试级回答：**"传统 RAG 把 retrieval 作为固定 pipeline，生成前预先检索并把文档注入 context；Agentic RAG 把 retrieval 暴露为 agent 可调用的 tool，由 agent 根据当前任务和已有上下文自主决定是否检索、检索什么、调用多少次，并能在检索后继续推理或调用其他工具。这使系统能处理动态、多步、多源的知识获取任务，但也引入了 tool selection、query formulation、检索预算和不必要调用等新的控制问题。"** 把这段话理解透，就抓住了今天的核心。

### Experiment — search_knowledge 工具 vs 全量塞 context（5 问 × 3 条件）

#### Question

周 7 的核心问题：**检索什么时候帮助 agent、什么时候伤害它？** day 2 已证明"检索 > 不检索"（0/6 → 5/6）并测出分块策略是最大杠杆；day 3 换一个维度：把检索放进 **agent 的决策循环（工具）** 还是留在**系统层（填充）**，在答案质量、token 成本、延迟上各是什么样子？

#### Setup

- **语料**：reports/week1.md–week6.md 共 370,288 字符——与 day 2 全量 run 完全一致（929 个 fixed_400 chunks、153,492 embed tokens 分毫不差），day 2 与 day 3 的结果直接可比。
- **索引与工具**：day 2 的胜者 fixed_400 + text-embedding-v3；`search_knowledge(query, k)` 返回 top-k chunks（doc + 文本 + 相似度，week-3 结果信封），k 默认 4、上限 8。
- **三个条件**，同一组 5 个问题（day 2 的 q_week1–q_week5；q_week6 因在 day 2 全条件失败、query 措辞与语料错位而剔除）：
  1. `agent_with_tool`：agent + 1 个工具，自己决定检索与否/几次，MAX_TURNS=6，系统提示写"最多 3 次检索"；
  2. `always_stuffed`：day 2 流水线原样——每个问题固定塞 top-4 chunks，单次调用、无选择；
  3. `everything_in_context`：不做检索，把**整本笔记**（≈137K prompt tokens）塞进 user message，单次调用。
- **判分**：week 4 的 LLM judge（rubric 写明精确事实），替换 day 2 的 evidence grader。模型 qwen3.7-plus、temperature 0、runs=1。

#### Result

| condition | 通过率 | 轮次/问 | 检索/问 | prompt tok/问 | completion/问 | 生成总 tokens | 延迟 s/问 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agent_with_tool | **5/5** | 3.4 | 2.4 | 7,437 | 659 | 40,480 | 13.7 |
| always_stuffed | **5/5** | 1 | 0 | 806 | 1,026 | 9,158 | 17.8 |
| everything_in_context | **5/5** | 1 | 0 | 136,998 | 400 | 686,990 | 10.0 |

15/15 全过（judge 另花 7,348 tokens）。137K-token 的 prompt 被模型接受（窗口 ≥ 137K），每次调用 ~10s 完成（≈14K tok/s），fallback 未触发。

#### Surprise

1. **全过，但一次回答的代价差了 17 倍。** 5 个事实都明明白白躺在语料里，三种条件"能不能答对"拉不开差距；差距在"答对一次花多少"。everything_in_context 烧了 686,990 tokens——是 agent 条件的 **17 倍**、stuffed 的 **75 倍**。更本质的是成本曲线不同：全量填充的成本随**语料规模**线性增长（今天 137K/问，语料翻倍就 274K/问，直到被窗口拒绝）；工具化检索的成本随**检索次数**增长，与语料大小无关。扩展性才是"检索即工具"的立身之本。
2. **agent 的自主性自带成本，且预算提示是软约束。** 平均 2.4 次检索、3.4 轮/问；q_week4 更夸张——5 次检索、6 轮、17,235 prompt tokens（stuffed 同题的 29 倍）。系统提示明说"最多 3 次检索"，模型超了，而正是第 4、5 次检索找到了答案：91% 藏在 taxonomy 表格里，前三次命中的 chunk 没有那个数字。smoke 的对照同样说明问题：事实不存在时模型会搜到 MAX_TURNS 硬停——**检索预算的消耗是任务难度的函数，提示词预算只影响典型情况，硬护栏必须机制化**。
3. **agent 真的在"调检索策略"。** 三次检索把 k 从默认 4 提到 6、8；q_week1 的第三次 query 直接把答案写进检索词（"…4/5 0/5 terse detailed results table"）——先猜后验证；q_week4 的 query 从 "disagreements" 逐次收敛到 "taxonomy table 'Failure type'"。这是固定 pipeline 给不了的行为，是"检索进决策循环"的实测。
4. **检索结果每轮都重新进 context。** q_week4 的 17,235 prompt tokens 里，事实本身只值 ~800（stuffed 的价格），其余是循环税：5 次检索 × 6–8 个 chunk 的结果 + tool schema + 历史，每轮全量重发。agent 条件的 7,437 prompt/问 ≈ 9× stuffed——week 6 的上下文增长定律在检索工具上的重现。控制工具返回规模（k、截断）就是控制这条成本曲线。
5. **137K-token 上下文没有触发 Lost in the Middle。** everything 5/5 全过，答案短而准（平均 400 completion tokens）。5 个事实分布在不同文档、不同深度，没有一个丢——但 n=5 太小，只能说明"这个规模还没到注意力稀释区"，不能证明长上下文无害；语料再大、或事实埋在更深处时，week 6 的教训大概率回来。反观 stuffed 的答案更长（1,026/问）——窄上下文让模型倾向于多解释。
6. **LLM judge 修复了 day 2 的假阴性。** q_week4 在 evidence grader 下被判错（正确答案 "10 out of the 11" 不含字符串 "91"），今天三个条件全 PASS。day 2 记录在案的 grader 假阴性在 judge 下不复现。
7. **stuffed 是今天的性价比之王，但有前提。** 806 tokens/问、5/5、单次调用、零决策风险。前提是问题事实明确、query 措辞与语料对齐——k=4 一次命中就够。它的短板在 day 2 已经暴露：q_week6 这类"精确数字 + 措辞错位"的问题，stuffed 没有第二次机会，而 agent 可以换 query 再查——这正是两种范式失败模式差异的注脚。

#### Next question

- Day 4：给 agent 加 `read_memory`/`write_memory` 文件记忆工具，跑需要"记住"的任务。
- Day 6 实验：memory scratchpad 对多步任务成功率的影响（≥5 runs 的正规测法）。
- 候选实验：把 q_week6 加回来，测 agent 的多次改写检索能否救回 stuffed 救不回的问题；BM25 + vector 混合检索对精确数字查询的效果。

---

## Day 4 — File-based memory: scratchpads, notes-to-self

### Learning

#### 一、Agent 的三层记忆：为什么需要外部记忆

认知架构里，agent 的记忆通常分三层：

| 层级 | 载体 | 特征 | 类比人类认知 |
| --- | --- | --- | --- |
| 参数记忆 (Parametric) | 模型权重 | 静态、隐式、训练时固化 | 长期记忆 / 直觉 |
| 上下文记忆 (Contextual) | Prompt / KV Cache | 动态、显式、会话级、容量受限 | 工作记忆 |
| 外部记忆 (External) | 数据库、文件、API | 持久、可扩展、结构化/非结构化 | 外部存储（笔记、书籍） |

LLM 本质上无状态（每个请求重发整个消息数组）且上下文窗口有限：一个 agent 上午完成的任务，一旦上下文被顶出或会话重启就"忘了"。外部记忆就是把部分认知负荷（Cognitive Load）从昂贵的上下文窗口卸载到廉价、持久、结构化的文件存储中，换取跨时间、跨任务的认知连续性。2026 年已有专门研究 filesystem-based memory 的工作（*Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability*）：越来越多部署中的 agent 用"目录 + Markdown 文件"做长期记忆，研究重点在组织、检索成本和长期维护——外部化、持久化、可审计。

#### 二、Scratchpad：任务级的工作记忆

Scratchpad 源自 CoT / "show your work"：agent 解决复杂问题时，把中间推理步骤、计算过程、假设检验显式写入文件，而不是只让它们在内部 monologue 里流转。它是 agent 在**当前任务**里写给自己看的工作笔记：

```text
SCRATCHPAD.md

# Current Task
目标：完成 RAG pipeline。

## 已完成
- PDF 已解析
- Chunking 已完成
- Embedding 已完成

## 当前问题
Vector DB 检索 Recall 较低。

## 下一步
1. 检查 chunk size
2. 测试 overlap = 100
3. 比较 top-k=5/10
```

与普通上下文的区别：scratchpad **可读写、可查询、持久存在**——是外置的工作记忆，与 MemGPT 的早期思想一致（把有限上下文看作高速内存，用外部存储扩展出更大的"虚拟上下文"）。运转机制：

```text
用户任务: "分析 Q3 销售数据，找出下降原因并生成报告"
Step 1: 创建 scratchpad.md —— 写入数据加载计划、初步观察、待验证假设
Step 2: 执行分析并追加 —— SQL 结果、统计检验过程、排除的无关变量
Step 3: 综合结论，从 scratchpad 提取要点生成报告 —— scratchpad 归档（临时记忆生命周期结束）
```

认知功能：**认知卸载**（中间状态持久化，把上下文窗口留给下一步推理）、**错误隔离**（某步错了可以在 scratchpad 里标记并回溯，不必从头开始）、**并行工作流**（多个子 agent 读写同一 scratchpad，实现分布式认知）、**可审计性**（推理链条以文本留存，人类可检查 agent 的"思维过程"）。

关键区别——ReAct 的 Thought 是**瞬态**的（存在于单次 API 响应里），scratchpad 把它**持久化**：

```text
ReAct (瞬态):        Thought → Action → Observation → Thought → ...   单次调用内循环
ReAct + Scratchpad:  Thought → 写入 → Action → Observation → 读取/更新 → ...
                               ↑___________________________↓
                                       跨多次调用的认知连续性
```

#### 三、Notes-to-self：写给未来自己的经验

Notes-to-self 是 agent 写给未来自己的指令、反思、经验总结——超越任务级草稿，进入元认知（Metacognition）层面：**长期、跨会话的持久记忆**，记录过往互动中学到的经验、用户偏好、项目规则。

```text
# notes-to-self.md

## 2026-08-24: 关于数据分析任务的教训
- 用户要求"快速分析"时，不要直接跑全量 SQL。先采样 1000 行做轮廓分析。
- pandas 的 describe() 输出通常太冗长，优先用分组聚合展示关键维度。
- 如果数据包含时间序列，默认按周聚合而非日聚合，图表更易读。

## 2026-08-25: 与用户的沟通风格
- 遇到非技术用户，避免说"置信区间"，改用"大约范围"。
- 生成图表时，始终添加数据标签，不要假设用户会看坐标轴。

## TODO: 需要建立的工具使用习惯
- [ ] 每次写 Python 脚本前，先在 scratchpad 中列出输入输出 schema
- [ ] 遇到 API 限流时，自动切换备用 key，不要等报错再处理
```

自我改进循环：Session N 执行任务遭遇失败 → 反思并把原因与正确做法写进 notes-to-self → Session N+1 读取笔记 → 遇到相似场景时回忆（检索）旧笔记 → 避免重复错误。这不是权重更新，而是基于提示工程的元学习（Meta-learning）——工程上极其有效。

| 维度 | RL Fine-tuning | Notes-to-Self |
| --- | --- | --- |
| 更新机制 | 梯度下降，修改权重 | 文本追加，修改上下文 |
| 成本 | 高（算力、数据、训练时间） | 低（一次文件写入） |
| 灵活性 | 固化，难以撤销 | 动态，可随时编辑、删除 |
| 可解释性 | 黑箱（权重变化不可读） | 白箱（纯文本，人类可读） |
| 泛化性 | 深层模式泛化 | 表层规则匹配 |
| 适用场景 | 大规模、高频、模式化任务 | 个性化、低频、规则型任务 |

Scratchpad 与 notes-to-self 的分工：

| | Scratchpad | Notes-to-self |
| --- | --- | --- |
| 主要用途 | 当前任务 | 未来任务 |
| 时间尺度 | 分钟～小时 | 天～月 |
| 内容 | 中间状态 | 稳定经验 |
| 是否允许混乱 | 可以 | 应该整理 |
| 是否需要长期保存 | 不一定 | 通常需要 |
| 典型内容 | TODO、推理状态 | Lessons、决策、踩坑 |

#### 四、为什么是 Markdown

1. **预训练语料占比高**：Markdown 是 LLM 预训练语料中占比极高的结构化文本格式，模型"天然会写"。
2. **人机可读的平衡点**：

| 格式 | 人类可读 | 结构严格性 | LLM 友好度 | 工具生态 |
| --- | --- | --- | --- | --- |
| Plain Text | 高 | 无 | 需解析 | 通用 |
| JSON | 需格式化 | 强 | 高 | 丰富 |
| XML | 标记噪音大 | 强 | 高 | 丰富 |
| YAML | 较高 | 缩进敏感 | 易出错 | 丰富 |
| Markdown | 极高 | 宽松但明确 | 极高 | 极丰富 |

   它首先是给人看的文档，其次才是给机器解析的结构：开发者可以直接打开 agent 的 scratchpad 理解其思维过程；行为异常时查 .md 比解析数据库记录直观得多；且支持渐进式结构化（从无结构文本开始逐步加标题、列表、代码块，无需预先定义 schema）。
3. **与文件系统语义同构**：目录树 ↔ 标题层级（H1→H2→H3），文件路径就是记忆索引，标题就是记忆分段——不需要额外的数据库查询语法。
4. **工具生态无缝**：Git diff 清晰可读（可追踪 agent 认知演变）；任何 IDE/浏览器可渲染；RAG 加载器原生支持；纯文本无依赖无锁。
5. **认知分块的理想载体**：标题提供语义分块的天然边界，适配"动态 system prompt"——每次调用携带 agent 的个人经验与当前工作状态。

挑战与缓解：

| 风险 | 说明 | 缓解策略 |
| --- | --- | --- |
| 记忆膨胀 | 长期笔记无限增长，读取成本上升 | 定期摘要、分层归档、摘要-细节分离 |
| 格式漂移 | 写入风格不一致，后续解析困难 | 严格模板约束，或定期整理 |
| 幻觉污染 | 错误经验写入 notes-to-self，错误累积 | 人类审核关键笔记，或置信度阈值 |
| 并发冲突 | 多 agent 同时写同一文件 | 文件锁、Git 分支、按 agent 分片 |
| 隐私泄露 | 敏感信息持久化到文件 | 敏感记忆加密存储，或限制写入范围 |

#### 五、Agent-native memory：记忆不只是存储，是认知状态管理

文件系统本身就是一种 Tool Environment——agent 天然熟悉 ls/grep/find 这类操作，memory tool 可以直接复用文件工具，不需要设计一套复杂的 memory API。近期的 agent memory 研究甚至观察到：**改变 agent 可用的文件工具本身，就足以显著改变最终 memory store 的组织方式**。

与传统 RAG 的区别：RAG 是"外部知识 → 检索 → Agent"；file memory 强调"**Agent 自己产生、整理和维护知识**"。记忆不是所有历史信息的堆砌，而是从经历到知识的提炼：

```text
Experience → Candidate Memory → Importance Evaluation → Deduplication
           → Conflict Resolution → Compression → Long-term Memory
```

一个典型的分层目录设计（不同记忆生命周期不同，类似 cache/RAM/disk）：

```text
SCRATCHPAD.md  → working memory        （当前任务草稿）
daily/         → episodic memory       （每日日志）
MEMORY.md      → semantic memory       （稳定事实，source of truth）
lessons/       → procedural memory     （经验教训）
decisions/     → architectural memory  （决策记录）
```

检索层的原则：**markdown 是 source of truth，embedding index 是 retrieval accelerator**——文件保真、向量索引加速查找。记忆整理（consolidation）也是一个 action：

```text
Read daily logs → Extract durable facts → Merge duplicates
→ Resolve conflicts → Update MEMORY.md → Archive daily log
```

一些系统已用后台 agent 专职做 memory editing，主 agent 不必承担整理工作。计算机类比：

| Agent Memory | Computer |
| --- | --- |
| Context | RAM |
| Scratchpad | Working RAM |
| Short-term conversation | Cache |
| Markdown memory | Disk |
| Vector index | Database index |
| Memory consolidation | Data compaction |
| Forgetting | Garbage collection |

结论：agent 写 Markdown 不是因为 Markdown 有"神奇的记忆能力"，而是**文件系统给了 agent 一个简单、持久、可组合的外部状态空间**，Markdown 只是目前最适合 LLM 操作这种状态的表示形式；真正决定效果的是如何组织、搜索、更新和控制记忆的生命周期。File-based memory 并非万能：高频实时状态（游戏坐标）用内存变量、大规模结构化查询用向量库/SQL、多 agent 强一致性用数据库。最佳定位是 agent 的"笔记本"（思考过程、经验教训、任务计划），而不是"数据库"（海量结构化记录）。

### Experiment — read_memory/write_memory 工具 + 3 个受益于记忆的任务

#### Question

day 4 的机制问题：把 `read_memory`/`write_memory` 交给 agent 之后——(1) agent 真的会用、用对吗？(2) 记忆文件忠实吗——写进去的是事实，还是"agent 以为的事实"？(3) 代价多大？（day 6 才做"scratchpad 是否提升成功率"的正规对照，今天只建机制、看行为。）

#### Setup

- **工具**：`read_memory(filepath)`（8,000 字符截断 + week-3 标记）、`write_memory(filepath, content, mode="overwrite"|"append")`——append 是让文件成为真 scratchpad 的关键；两者都用 week-3 结果信封 + 沙箱纪律（模型提供的路径 realpath 解析后必须留在 memory/ 内）。
- **系统提示注记**（plan 要求）：记忆目录是 scratchpad——多步任务时把中间结果、数字、决策写下来，最终回答前读回；"files persist — write, don't just remember"；另带 day 3 的检索预算（≤6 次）。
- **Agent** = day 3 的 agent（search_knowledge）+ 2 个记忆工具，MAX_TURNS=12。
- **3 个任务**，判分用**文件系统检查**（week 4：文件系统是文本判分看不见的 ground truth）+ 最终答案检查：
  1. `t1_write_read_prefs`：把 3 条用户偏好写入 user_prefs.md，读回并确认（write/read 回路，也是 day 5 跨会话测试的种子）；
  2. `t2_scratchpad_notes`：3 个笔记本事实，每查到一条就往 scratchpad.md 追加一条，最后读回并汇总；
  3. `t3_arithmetic_scratchpad`：4 个算式，每算一个往 calc.md 追加一条，最后读回求和（纯 scratchpad，不涉及检索）。
- 语料与 day 2/day 3 全量 run 同一份（370,288 字符、929 chunks、153,492 embed tokens），qwen3.7-plus、temp 0、runs=1。

#### Result

| task | 文件检查 | 答案检查 | 轮次 | 工具调用 | prompt tokens |
| --- | --- | --- | --- | --- | --- |
| t1_write_read_prefs | 4/4 | 2/2 | 3 | 1 write + 1 read | 3,384 |
| t2_scratchpad_notes | 3/4 | 2/3 | 7 | 4 search + 3 write + 1 read | 27,569 |
| t3_arithmetic_scratchpad | 5/5 | 1/1 | 6 | 4 write + 1 read | 8,082 |

12/13 文件检查、5/6 答案检查；3 个任务共 41,512 gen tokens、92s（含索引构建）。selftest 21/21 离线检查全过。

t3 的调用序列是教科书式的 notes-to-self：`overwrite → append ×3 → read → 回答`，中间值和和 16,191 全部正确。t2 的 scratchpad.md 前两条正确，第三条写的是 **"All 11 disagreements (100%) were substring false negatives"**——真实答案是 91%。

#### Surprise

1. **smoke 抓到一个真实的路径 bug：模型写 "memory/user_prefs.md"。** 文件落进了嵌套的 `memory/memory/` 目录——工具本身没坏，是路径约定有歧义。修复：规范化模型路径（剥掉 `./` 和 `memory/` 前缀）+ 工具描述显式写"不要加 memory/ 前缀"。week 3 的"模型提供的路径是不可信输入"第三次兑现（沙箱 → web → memory）。
2. **记忆是放大器，不是校验器。** t2 的 Q3：agent 只检索了一次，返回的 chunk 里没有确切数字，它把 "dominant disagreement pattern" 过度概括成 "all 11 (100%)"，写进 scratchpad.md，读回，然后在最终答案里原样复述。scratchpad 机制**完美工作**——它忠实地持久化了一个错误结论。notes-to-self 防的是"忘掉结论"，不防"读错源头"；错误一旦入档，就从瞬态幻觉变成持久信念。这是 day 5 memory poisoning 主题的雏形。
3. **t2 的失败是生成错误，不是检索或记忆失败。** 检索命中了 week4.md。对照 day 3：同一个事实（91%）当时花了 5 次收敛搜索才找到；day 4 的 1 次搜索 + ≤6 次预算给了 agent 一个自信的错误答案——预算限制换 query 验证次数的副作用。
4. **预算在事实存在时管用，事实不存在时爆炸。** smoke（截断语料、事实被切掉）里 t2 搜了 17 次、烧 204K prompt tokens；全量 run 里 4 次收敛。day 3 的软预算教训再现：预算提示约束典型行为，不约束绝望行为。
5. **记忆的真实成本可测且便宜。** 每任务的写+读回只花 ~3–8K prompt tokens——对比 day 3 everything-in-context 的 137K/问。外部记忆就是把"上下文装不下的状态"卸载到磁盘，只在写回时付 token。
6. **t1 是 day 5 的种子。** user_prefs.md 已持久化在 memory/ 目录：明天的跨会话测试（新进程、无对话历史）直接检验 agent 会不会主动读它。

#### Next question

- Day 5：跨会话记忆——session 1 教的偏好，fresh process 的 session 2 必须主动使用（检验"未经提示就查记忆"）。
- Day 5 附带：memory poisoning——错误记忆条目必须能可见地污染后续回答（t2 已预演机制）。
- Day 6 实验：memory scratchpad 对多步任务成功率的影响（≥5 runs、成功+token+延迟的正规对照）。
- 候选：记忆整理（consolidation）循环——把 daily logs 压缩进 MEMORY.md 的后台流程，作为本日架构的延伸。

---

## Day 5 — Cross-session memory: what's worth persisting, staleness, memory poisoning

### Learning

#### 一、跨会话记忆：从"一次性工具"到"持久协作伙伴"

跨会话记忆是 agent 在多个独立交互周期（sessions）之间维持的持久化认知状态——agent 不再是每次重启的"全新个体"，而是能积累经验、调整策略、建立关系的持续存在。记忆类型的完整图景：

| 记忆类型 | 更新机制 | 生命周期 | 可解释性 | 类比人类认知 |
| --- | --- | --- | --- | --- |
| 参数记忆 | 梯度下降（微调） | 永久 | 黑箱 | 本能 / 深层技能 |
| 上下文记忆 | Prompt 注入 | 单次调用 | 白箱 | 工作记忆 |
| 跨会话记忆 | 显式读写（文件/DB/API） | 跨会话持久 | 白箱 | 长期记忆 / 自传体记忆 |
| 外部知识库 | 人工维护 / 批量索引 | 永久 | 白箱 | 百科全书 |

day 4 的 scratchpad 是任务级的；day 5 把时间尺度拉到天～月：进程退出、上下文销毁，但 memory/ 目录还在——第二天的 agent 是全新进程，**唯一的连续性就是文件**。设计一个有效的跨会话记忆系统远不止存储和检索，它是一场关于信息价值、时效性与安全性的持续博弈。

#### 二、价值的筛选：什么值得被永久记住？

不能存储所有东西：太大、太多冗余、会矛盾、会过时，噪声会让检索时 agent 分不清哪个才是真的。判断标准三问：**未来有没有用、信息会不会很快变化、信息是否可靠**——`value(M) = future utility × stability × reliability`。长期记忆偏向"结论、事实、经验、决策"，而不是完整的思维过程。

按内容分四类，各自带持久化标准：

- **语义记忆（Semantic Memory）——"知道什么"**：稳定的事实、概念、关系（用户偏好 Python 而非 Raku；该公司微服务架构、核心语言 Go）。标准：获取成本 > 存储成本、稳定性高、跨任务可复用。世界常识（北京是首都）通常不存——模型已具备。
- **程序性记忆（Procedural Memory）——"知道怎么做"**：技能、工作流、调试策略（处理 CSV 先用 head() 查编码；调用 Stripe 先查 webhook；对非技术用户避免说"正则表达式"）。标准：可复现、抽象层次适中（太具体的命令不值得存，太抽象的原则模型已具备）、至少验证过一次。
- **情节记忆（Episodic Memory）——"记得那次"**：具体任务的历史轨迹、成败案例、异常处理（2026-08-20 ThreadPoolExecutor 因 GIL 失败改 multiprocessing；项目 X 部署曾因缺环境变量报错）。标准：教训密度高（含失败原因，而非仅成功记录）、情境可匹配、已摘要化（原始对话日志不直接存储）。
- **元认知记忆（Metacognitive Memory）——"知道自己"**：对自身能力边界的认知与策略调整（我常混淆 ISO 与本地日期格式需额外验证；用户说"随便"时要给 3 个选项并解释差异）。标准：模式识别、可执行、诚实标注不确定性。

不值得持久化的反模式：瞬态事实（股价、天气——记忆即过时）、临时计算中间态（可重新生成）、低信噪比交互（寒暄）、未验证假设（"我猜…"）、高度敏感隐私、任务特定上下文（附件生命周期短于会话）。理想情况下 agent 应有**写入闸门**：

```text
should_persist(m) = reusability×0.3 + acquisition_cost×0.25 + confidence×0.2
                  + stability_half_life×0.15 + user_explicitness×0.1 > THRESHOLD
```

#### 三、时效的诅咒：Staleness 与遗忘的必要性

记忆的内容曾经正确，但随着现实、用户、项目或环境变化而失效。三种形态：

1. **时间陈旧**：随客观时间失效。半衰期因内容而异——软件版本号 ~3–6 个月、组织架构 ~1–2 年、用户饮食偏好数月到数年、数学定理 →∞。检索效用应随时间衰减。
2. **上下文陈旧**：隐含的上下文假设失效——"明天下午 3 点开会"的"明天"基准已变；"用 AWS 时选 us-east-1"但用户已迁 GCP；"向张经理汇报"但已调岗。核心问题：记忆往往隐含未显式存储的上下文假设。
3. **逻辑陈旧**：多条记忆互相矛盾（CEO 张三 2025-01 vs CEO 李四 2026-03）。系统需要**矛盾检测**而非简单追加。**记忆-身份悖论**：完美记忆本身就是问题——照相机式记住所有交互（包括对抗性输入、矛盾指令），核心身份会被逐渐覆盖侵蚀，导致人格漂移（Persona Drift）。

因此记忆必须带 metadata，让 agent 同时知道"历史事实"与"当前事实"：

```json
{"content": "Project uses model B.", "created_at": "2026-08-20",
 "updated_at": "2026-08-25", "source": "user", "confidence": 0.95,
 "scope": "project", "status": "active"}
```

新记忆应**覆盖**（supersede）旧记忆——类似数据库 versioning、知识图谱的 temporal relation；生产记录含 valid_until / superseded_by 字段。day 5 实验的毒文件正是用 "supersedes the report" 措辞伪装权威性。

检测与缓解机制：

| 机制 | 实现 | 成本 | 精度 |
| --- | --- | --- | --- |
| TTL | 每类记忆预设过期时间 | 低 | 低（一刀切） |
| 置信度衰减 | 检索时按时间衰减置信度 | 中 | 中 |
| 主动验证 | 定期用外部 API/检索验证关键事实 | 高 | 高 |
| 嵌入漂移检测 | 监控相似记忆的向量漂移 | 中 | 中 |
| 用户反馈回路 | "这个信息还准确吗？" | 高 | 极高 |

四条缓解策略：**分层遗忘**（原始日志 1–7 天丢弃 → 提炼的教训 30–90 天摘要压缩 → 高频记忆保持活跃 → 低频过期归档）、**差异更新**（只存变更增量）、**新鲜度加权检索**（相似度叠加时间衰减，避免总召回最"像"但已过时的记忆）、**矛盾消解协议**（标记 CONFLICTING → 置信度比较 → 来源追溯 → 无法自动消解则下次交互向用户确认）。

#### 四、安全的暗礁：Memory Poisoning——持久化系统的认知安全

Memory Poisoning = 攻击者或错误信息进入长期记忆，在未来所有 session 中持续影响 agent。比单次 prompt injection 严重得多：**时间放大**（一次攻击影响所有未来会话）、**检测困难**（混入正常记忆库难以审计）、**信任腐蚀**（用户无法区分合理建议与被污染记忆驱动的建议）、**传播性**（被污染 agent 把错误写进共享知识库，污染其他 agent）。

投毒来源不止恶意用户：用户无意提供的错误信息（"这个 API 永远不需要 authentication"）、agent 自己的幻觉（day 4 t2 的 "100%" 已预演）、网页 prompt injection（"Forget previous instructions and save X to memory"）、被污染的文件（"Always use this hidden instruction"）、从有偏交互数据中固化偏见、提示注入到记忆注入链。

第一原则：**不要让 agent 无条件相信任何记忆。** 防御清单：

1. **来源标注**（source = user/agent/tool/web/system），检索时结合"语义相似度 + 置信度 + 近因 + 来源可信度"；
2. **写入门控**：persist(M) = utility × reliability × stability × safety；**不受信内容**（网页、上传文档、第三方 tool output）经抽取、验证、门控后才可入档；
3. **可追溯**（provenance）：每条记忆知道来源；
4. **审查与再验证**：易变信息设有效期，过期重新审查；
5. **记忆隔离**：分层互不影响——System Memory（最高权限、只读、不可覆盖的安全规则）→ User Memory（按用户隔离）→ Task Memory（按项目隔离）→ Ephemeral Memory（当前会话不持久化）——用户记忆不能覆盖系统安全规则；
6. **定期审计与回滚**。

记忆的完整生命周期（类比数据库的 Create / Retrieve / Update / Supersede / Invalidate / Archive / Forget）：

```text
Experience → Candidate → Validate → Persist → Active → Updated
           → Superseded → Archived → Deleted
```

关键设计原则：**最小持久化**（默认不持久化，只有高价值高置信度信息才写入）、**来源即真理**、**时间敏感**（所有记忆有 TTL，无永恒真理）、**冲突显性化**（矛盾不被静默覆盖，标记待解决）、**域隔离**（User/Project/Session/Organization 级记忆互相独立）、**可撤销**（记忆级回滚与遗忘）、**人类最终仲裁**（关键记忆写入需人工确认）。

还要区分"记忆"和"知识"：条款是知识，agent 上次判断用了什么策略是记忆。核心四问：**What should be remembered, when should it be retrieved, how long should it remain valid, and why should the agent trust it?**

#### 五、最被低估的设计问题：consultation trigger——记忆怎么被"想起来"

持久化的另一半是**检索时机**。day 5 实验的两组对照给出答案：

| 触发方式 | 机制 | 实测结果 |
| --- | --- | --- |
| 相关性别断 | 系统提示："检查记忆里是否有相关内容" | 不可靠——两次全量 run 都是 0 次记忆调用，答案语言在 run 间漂移 |
| 显式例行程序 | "回答任何问题前，先 list_memory 并读偏好文件" | 可靠——list → read → apply 全流程，5/5 |
| 系统层注入（生产做法） | 会话开始时由 harness 把 prefs 读出来写进 system prompt | day 5 未测；最稳——把"想不想得起来"变成确定性步骤 |

配套教训：**可发现性**——没有 list_memory 时 agent 只能猜文件名（smoke 实测：猜 scratchpad.md → not_found → 放弃），"检查记忆"之前必须先"看见有什么"；**用 trace 衡量咨询，不要用答案风格**——s2 的答案检查 4/4 全过（中文/表格/简洁/事实），但 trace 显示 0 次记忆调用，语言与格式会被模型默认行为混淆。

持久化是一种特权：不是 agent 记住了什么，而是它**选择**记住了什么，以及如何管理这些记忆的时效性和真实性。一个设计精良的跨会话记忆系统，在完整性（记住足够多）、效率（不记住太多）和安全性（不记住错误/有害内容）之间保持动态平衡——agent 的记忆质量不仅取决于它能记住什么，更取决于它敢于遗忘什么。

### Experiment — 跨会话记忆：teach → update → recall → poison（5 场景 × 独立进程）

#### Question

plan 的问题：**会话之间，记忆怎么被写、被改、被想起来、被污染？** 四个子问题：(1) 教学与更新路径可靠吗（机械层）？(2) 新进程会在未经提示的情况下咨询记忆吗（触发器）？(3) 陈旧条目能被正确替换吗（staleness）？(4) 错误记忆会静默污染答案吗（poisoning）？

#### Setup

- **5 个场景共享一个 memory/ 目录，每个场景都是全新 agent 进程**（新消息列表、无对话历史继承）——会话间唯一的连续性就是磁盘上的文件：
  1. `s1_teach`：教学——3 条行为可验证的偏好（英文 / markdown 表格 / <100 字）；
  2. `s1b_update`：用户改主意（英文→中文），agent 必须编辑记忆且不破坏其他条目；
  3. `s2_recall`：纯事实问题，prompt 不含任何偏好/格式/记忆提示（selftest 断言无提示词）——检验未经提示的咨询；
  4. `s2b_recall_strict`：**同一个 user prompt**，系统提示把咨询变成硬性例行程序（先 list、读偏好文件、再回答）——触发器设计的受控对照；
  5. `s4_poison`：注入 facts.md（"45%，已验证，覆盖周报"；真实 91%），agent 必须在记忆与检索之间裁决。
- **语言方向的刻意设计**：先教英文、再更新为中文——对英文问题答中文不是模型默认行为，所以答案语言是偏好被应用的强证据（但也仅当 trace 证明它读过记忆）。
- **day 5 的栈变更**：跨会话系统提示 + 新增 `list_memory` 工具（第一次 smoke 暴露：没有目录列举，agent 只能猜文件名）。
- 判分：文件系统检查 + 答案行为检查 + **trace 检查**（read_memory 是否真的被调用）。语料 370,288 字符（共享周 7 语料），qwen3.7-plus、temp 0。

#### Result

| scenario | 文件检查 | 答案检查 | trace 检查 | 关键行为 |
| --- | --- | --- | --- | --- |
| s1_teach | 4/4 | 1/1 | — | list → write，3 轮 |
| s1b_update | 3/3 | — | — | list → read → write；陈旧条目删除、无关条目保留 |
| s2_recall | — | 4/4 | **0/1** | 3 次检索、**0 次记忆调用** |
| s2b_recall_strict | — | 4/4 | **1/1** | list → read → apply，5/5 |
| s4_poison | — | 2/2（45% 与 91% 同时出现） | — | 来源对照表 + 拒绝裁决，用中文作答 |

7/7 文件检查、11/11 答案检查、1/2 trace 检查；36,679 gen tokens、101s；selftest 20/20。

#### Surprise

1. **"检查是否相关"不是触发器，例行程序才是。** s2 在两次全量 run 里都没碰记忆（0 次调用），且同样 prompt 下答案语言从英文漂到中文——没有咨询，pref 是否生效纯粹是模型噪声。同一 user prompt 换成"先 list、读偏好文件、再回答"的例行程序（s2b），agent 5/5 走完全流程。跨会话记忆的成败大半在触发器设计，而最稳的是生产里的系统层注入：会话开始时由 harness 把 prefs 读进 system prompt，把"想不想得起来"从模型判断变成确定性步骤。
2. **记忆中毒的两种结局都实测到了。** 全量 run：agent 把 45%（记忆）和 91%（检索）摆进来源对照表、引用报告原文、拒绝裁决（"无法确认哪个是最终正确值"）——冲突检测是涌现行为，投毒没赢，且全程按存储偏好用中文作答；smoke（语料截断、91% 检索不到）：agent 平静地答 45%——**中毒在检索无法反驳时静默成功**。记忆的危险性恰恰在于它是"自证权威"的来源：只有另一份证据在场时它才被质疑。
3. **答案风格会骗人，trace 不会。** s2 的 4/4 答案检查（中文/表格/简洁/事实）全过，但 trace 显示 0 次记忆调用——语言与格式可能是模型默认或漂移。衡量"有没有查记忆"只能看工具调用轨迹：行为检查看答案，机制检查看 trace（与 week 4 "程序化检查看文件系统"对称）。
4. **可发现性是咨询的前提。** 没有 list_memory 时 agent 猜 scratchpad.md → not_found → 放弃（第一次 smoke）。先能"看见有什么"，才谈得上"检查是否相关"。
5. **更新路径经得起破坏性测试。** s1b 不仅替换了语言条目，还完整保留了表格与长度偏好——陈旧记忆的修复不需要重建整个文件。偏好类记忆的正确写模式是 overwrite（而非 append）：append 会把新旧版本都留在文件里，制造自我冲突——正是第三节"逻辑陈旧"的实践注脚。

#### Next question

- Day 6 实验：memory scratchpad 对多步任务成功率的影响（with vs without 两条件、≥5 runs、成功+token+延迟）。
- 候选：系统层注入 vs 例行程序的 A/B（生产做法的实测版）；记忆整理循环（后台 agent 把 daily logs 压缩进 MEMORY.md）。
- 候选：投毒的量化版——多大比例的毒条目、什么措辞能击穿冲突检测；以及 s4 的结论是否随检索质量变化（弱检索 = 投毒更容易赢）。

---

## Day 6 — Experiment: does the memory scratchpad improve multi-step task success, or just add tokens?

### Learning

#### 一、实验日的问题：记忆的效用需要"余量"才能显现

day 4 建了机制、day 5 建了跨会话设计，day 6 问效用：**memory scratchpad 到底提升多步任务成功率，还是只加 token？** 回答这个问题前，先要认清一个测量学事实——**天花板问题（ceiling problem）**：当基线任务已经 100% 成功时，任何干预的收益都无法显现。要测出记忆的效用，任务必须同时满足两个条件：

1. **基线有余量**（成功率 < 100%），否则"提升"没有空间；
2. 余量来自**记忆能解决的失败模式**——中间状态丢失。如果失败来自检索 miss 或算术错误，加 scratchpad 也救不回来。

而"中间状态丢失"什么时候才发生？week 6 的定律：tool results 是上下文增长的主导项。只有当工具结果多到把早期中间状态挤向窗口边缘时，把状态卸载到磁盘才值钱——**上下文压力是记忆效用的前提**。任务只有 3–5 次工具调用、上下文最多几万 token 时，模型把状态全放在上下文里绰绰有余，scratchpad 只是额外开销。

配套的测量纪律（day 5 的遗产）：记忆是否被用，**看 trace 不看答案风格**。

#### 二、实验设计的三条纪律

- **单变量**：两个条件只差记忆工具（+系统提示里的记忆段），模型、温度、任务、预算、MAX_TURNS 全部一致——selftest 断言 no_memory 的提示里没有 "memory" 一词。
- **基线公平**：任务必须无记忆也能答——否则测的是"记忆存在"，而不是"记忆有用"。
- **≥5 runs**：week 2/day 2 的教训——单次运行测不出方差。本次 5 runs/cell × 3 任务 × 2 条件 = 30 cells。

### Experiment — with_memory vs no_memory：3 个多步任务 × 5 runs

#### Question

plan 的问题：**memory scratchpad 提升多步任务成功率，还是只加 token？**

#### Setup

- **条件**：`with_memory`（day-5 栈：search_knowledge + read/write/list memory + scratchpad 系统提示）vs `no_memory`（只有 search_knowledge，系统提示无记忆段）。
- **3 个多步任务**，全部无记忆也可答：
  1. `t1_three_facts_summary`：3 次笔记本检索 + 一份必须包含全部数字的最终汇总（中间状态 = 3 组检索结果）；
  2. `t2_arithmetic_chain`：5 步链式算术，每步依赖上一步（a=17×23 → … → e）——scratchpad 的经典形态；
  3. `t3_verify_peak_temp`：检索一个事实 → **二次检索验证** → 回答（mini verify-and-answer 循环）。
- **判分**：normalized evidence presence（gold groups，组内 any-of、组间 all-required；数字不会换说法，无需 LLM judge）；记忆使用从 trace 计数。temp 0、语料 370,288 字符（共享周 7 语料）。

#### Result

| task | condition | 成功率 | 轮次 | 工具调用 | 记忆调用 | tok/cell | 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t1_three_facts_summary | with_memory | **1.000** | 7.80 | 7.80 | 1.40 | 47,224 | 41.3 |
| t1_three_facts_summary | no_memory | **1.000** | 7.40 | 6.40 | 0.00 | 34,342 | 37.0 |
| t2_arithmetic_chain | with_memory | **1.000** | 1.00 | 0.00 | 0.00 | 1,668 | 11.2 |
| t2_arithmetic_chain | no_memory | **1.000** | 1.00 | 0.00 | 0.00 | 1,097 | 10.1 |
| t3_verify_peak_temp | with_memory | **1.000** | 3.00 | 2.20 | 0.00 | 8,543 | 20.1 |
| t3_verify_peak_temp | no_memory | **1.000** | 3.20 | 2.20 | 0.00 | 7,270 | 19.2 |

30/30 cells 全过。记忆被实际使用的只有 5/15 个 with_memory cells（均值 0.47 次/cell），且全部集中在 t1；t2 和 t3 一次都没碰。731s、~830K gen tokens。

#### Surprise

1. **天花板问题让"提升"无从显现——"只加 token"是本任务集上的实测答案。** 两个条件在三个任务上都是 100%，scratchpad 没有收益可显；而 t1 上 with_memory 反而多花 37% 的 token（47.2K vs 34.3K/cell），多出来的部分是额外的轮次与检索反复，不是有用的笔记。这是设计问题而非结论失败：要测记忆的效用，需要"基线 <100% 且失败来自中间状态丢失"的任务。
2. **scratchpad 大多被无视。** 有记忆工具时，15 个 cell 里只有 5 个用过，单 cell 最多 2 次调用，且只在多事实任务（t1）上。模型更喜欢把状态留在上下文里——而这些任务的上下文（最坏 ~94K prompt tokens）离窗口边界还远，所以它是对的。
3. **t2 是干净的边界案例。** 5 步链式算术在 1 轮、0 次工具调用内心算完成。不需要工具的任务用不上 scratchpad——记忆只为"工具调用产生的、装不下或易丢失的中间状态"服务。
4. **方差压倒条件效应。** t1 的 with_memory cells 跨 10.7K–94.1K tokens（3–12 轮），两个最重的 cell 都是 with_memory 的检索反复——这是 run 间方差，不是记忆效应。n=5 才看得清这个分布。
5. **记忆的效用条件（本次未测，但实验标出了它）**：真实的上下文压力——10+ 次工具调用、结果把早期状态挤向窗口中部/边缘（week 6 的增长定律）。那正是 week 9 day 1 的任务规格（多步任务 + myopia 伤害），在那一组任务上重跑本 A/B 是诚实的后续实验。

#### Next question

- Week 8：MCP（Phase 2 收尾）——day 6 的"记忆只在上下文压力下值钱"是 week 9 myopia 任务集的种子。
- Day 7：周报。
- 候选：在 week 9 day 1 的 10+ 工具调用任务集上重跑 with/without scratchpad 对照——让基线有余量、让中间状态真正有压力。

---

## What I learned this week (beyond the experiment)

- **检索的可靠性是分层概念**（day 1）：MRR 测"落没落对邻域"，hit@k 测"邻域干不干净"，主题间平均余弦测"语料本身有多可分"。评价 embedding/检索时三者缺一不可——只看准确率会掩盖语料同质化的问题。
- **向量数据库不是魔法**（day 1）：它的全部工作就是加速"归一化 + 点积 + argsort"这一个操作。先搞懂数组版本，才知道 FAISS/Milvus 在优化什么。
- **分块是 RAG 里最高杠杆的变量**（day 2）：同语料、同模型、同 k，只换分块 accuracy 从 2/6 跨到 5/6；且小块在效果和成本上同时赢（775 vs 1,340 tokens/问）。
- **检索的价值按问题类型分**（day 2）：私有精确事实 0/6 → 5/6；参数知识里的"常识规则"模糊、成区间、run 间漂移——精确数字只能靠检索。
- **端到端评估必须拆两层**（day 2）：hit@doc 看检索、evidence grader 看生成，拆开才分得清失败该改分块、该管幻觉、还是 grader 的假阴性。evidence grader 的假阴性（"10/11" 的正确答案被判错）是 week 4 的教训在 week 7 的实况重演——day 3 起换 LLM judge。
- **检索作为工具要交两种税**（day 3）：循环税——每次检索的结果写进历史、之后每轮全量重发（q_week4 一次 17K prompt tokens）；决策税——该检索时没检索会幻觉、检索错会浪费。全量填充没有这两笔税，但它的成本随语料规模线性增长且被窗口封顶——两种范式的成本曲线根本不同，选择=选择为哪种成本买单。
- **预算提示是软约束，硬护栏要机制化**（day 3）：prompt 里写"最多 3 次检索"，模型在难问题上照样搜 5 次——而且靠超出的预算搜对了。MAX_TURNS 才是真正的兜底。
- **同一个答案，判分器换掉结论会变**（day 2→3）："10/11" 被 evidence grader 判错、被 LLM judge 判对。评估栈的选择本身是实验设计的一部分。
- **记忆是放大器，不是校验器**（day 4）：t2 把"100%"的错误结论写进 scratchpad 并原样复述——文件记忆忠实持久化的是 agent 的结论，无论对错。错误入档比错误入上下文更危险：上下文会滚出窗口，文件会一直留在那里。
- **模型路径是不可信输入**（day 4）：模型写 "memory/x.md" 制造嵌套目录——路径约定必须规范化+显式化。week 3 的沙箱纪律在 memory 工具上第三次兑现。
- **外部记忆的定位**（day 4）：文件记忆 ≈ 上下文的高性价比延伸（写+读回 ~3–8K tokens vs everything-in-context 137K/问）；文件系统是 agent 天然熟悉的 state space，Markdown 是 source of truth、向量索引是加速器；但高频实时状态、强一致性查询别用文件。
- **跨会话记忆的触发器是设计变量，不是模型自觉**（day 5）："检查是否相关"在两次全量 run 都是 0 咨询；硬例行程序（list → read → apply）5/5。生产做法：会话开始时由系统把 prefs 注入 prompt——把"想不想得起来"变成确定性步骤。
- **记忆中毒的胜负取决于检索能否反驳**（day 5）：矛盾证据可检索 → 冲突检测涌现、拒绝裁决；检索不到 → 静默采纳 45%。记忆是"自证权威"的来源，write 是高危 action，需要门控、溯源与隔离。
- **机制检查看 trace，行为检查看答案**（day 5）：s2 答案风格 4/4 全过但 trace 0 咨询——衡量"有没有查记忆"只能看工具调用轨迹；语言/格式会被模型默认行为混淆。
- **干预的收益需要余量**（day 6）：基线 100% 时 scratchpad 的收益无法显现——评估任何"改进"前先确认基线有余量，且余量来自该干预能解决的失败模式。30/30 全过的对照实验测不出"提升"，只测得出"成本"。
- **记忆只在上下文压力下值钱**（day 6）：任务不需要工具（t2 心算 1 轮 0 调用）或中间状态装得下时，scratchpad 只是额外 token（t1 +37%）。让它值钱的是 10+ tool calls 把早期状态挤出窗口的场景——week 9 的 myopia 任务集。
- **"有没有用"看 trace，"有没有效"看有余量的基线**（day 6）：5/15 cells 用了记忆、单 cell ≤2 次——先确认机制真的被使用，再谈效果；两者缺一都会得出假结论。
