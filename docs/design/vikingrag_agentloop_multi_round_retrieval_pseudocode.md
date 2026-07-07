# VikingRAG 多轮 Agentic Retrieval 具体流程

本节用于论文中 “Specifics of the multi-round agentic retrieval process of VikingRAG” 部分。前文已经说明了 VikingRAG 的整体动机：相比一次性检索或固定的 retrieve-then-read pipeline，VikingRAG 将检索过程交给 agent 逐轮决策。每一轮中，agent 可以根据已有证据决定下一步是扩大搜索、读取某个目录片段的原文、还是用 grep 精确定位数字和短语。

VikingRAG 的关键不是简单地“多调用几次检索器”，而是把外部知识库组织成可随机访问的目录片段。语义检索先返回一组相关 URI；随后 agent 可以像访问列表元素一样，按需读取某个 URI 对应的正文片段，或在某个 URI 范围内执行正则检索。也就是说，VikingRAG 的多轮过程是一个“search -> random access read/grep -> reflect -> next action”的闭环。

本文以 `/home/zhanggaoyuan.225/vikingrag/benchmark/RAG/config/financebench/financebench_bot_config.yaml` 对应的普通 bot 实验为参照。该设置代表本文要描述的基础多轮检索路径，不包含 reference/relation 边扩展，也不包含答案后建边。

## 运行范围

在该配置中，关键参数如下：

```yaml
vikingbot:
  max_iterations: 15
  search_limit: 10
  log_tool_calls: true
  enable_linking: false
```

这意味着一次问题回答最多进行 15 轮 agent 迭代。每一轮由 LLM 决定是否调用工具。如果调用工具，工具结果会加入上下文，下一轮 LLM 再基于这些结果判断是否继续搜索、读取原文、grep 精确匹配，或直接生成答案。由于 `enable_linking: false`，本实验不执行答案后的建边逻辑；由于该配置没有 `use_relations: true`，搜索阶段也不做关系边扩展。

## 贯穿例子

为了让伪代码更容易嵌入论文叙事，下面使用一个 FinanceBench 风格的例子说明每一步：

> 问题 `q`: “What was Company X's capital expenditure in fiscal year 2022?”

在这个例子中，agent 首轮可能调用 `openviking_search(q)` 搜索相关财报目录片段；如果搜索结果只给出摘要和 URI，agent 下一轮会调用 `openviking_multi_read(U)` 读取最相关的财报正文；如果正文很长，agent 还可以调用 `openviking_grep(u, ["capital expenditure", "capital expenditures"])` 精确定位包含目标数值的行。每次工具结果都会追加到上下文 `M` 中，下一轮 LLM 再基于新证据决定是否继续。

## 符号与变量说明

本文伪代码采用论文中的抽象符号，不直接照搬实现中的变量名。

| 符号 | 含义 |
|---|---|
| `q` | 用户问题，也是语义检索时的查询文本。 |
| `M` | LLM 消息上下文。初始时由系统提示和包含检索指令的用户问题组成；后续轮次会追加工具动作、工具结果和反思提示。 |
| `T` | 单次回答允许的最大 agent 迭代轮数；在示例配置中为 15。 |
| `A` | 固定的工具集合，包括 `openviking_search`、`openviking_multi_read`、`openviking_grep` 等。 |
| `L` | 语言模型，用于选择工具、反思检索结果和生成最终答案。 |
| `a` | 最终答案。 |
| `t` | 当前 agent 迭代轮次。 |
| `R_t` | 第 `t` 轮 LLM 响应。 |
| `g_t` | 第 `t` 轮 LLM 选择的检索动作，例如搜索、读取或 grep。 |
| `o_t` | 执行检索动作 `g_t` 后得到的观察结果。 |
| `u` | OpenViking 搜索或读取的目标 URI 范围。 |
| `k` | 期望返回的候选文档数量；在示例配置中搜索数量为 10。 |
| `V` | OpenViking 客户端抽象。 |
| `B` | 实际执行语义检索的后端搜索器。 |
| `X` | 后端搜索器返回的原始结果。 |
| `D` | 当前候选文档集合。 |
| `d` | 单个候选文档。 |
| `Y` | 返回给 LLM 的格式化工具结果文本。 |
| `U` | 待读取的 URI 列表。 |
| `m` | 并发上限；代码中读取和 grep 默认最多 10 并发。 |
| `Q` | 异步任务集合。 |
| `Z` | 异步任务完成后的结果集合。 |
| `z` | 单个异步任务结果。 |
| `p` | 单个正则检索模式。 |
| `P_g` | grep 使用的正则模式集合。 |
| `c` | grep 是否忽略大小写的布尔开关。 |
| `G` | grep 结果映射，键为 URI，值为命中行集合。 |
| `n` | grep 命中总数。 |
| `match` | 单条 grep 命中记录。 |
| `matches` | 某个正则模式返回的一组命中记录。 |

## 初始上下文 M

在 benchmark 的 VikingBot 路径中，系统通过 `vikingbot chat -e` 进入 eval 模式。此时初始上下文 `M` 不包含历史对话、workspace bootstrap、skills 或用户记忆，而是由两条消息组成：

```text
M_0 = [
  {role: "system", content: P_sys},
  {role: "user", content: P_ret(q)}
]
```

其中系统提示 `P_sys` 为：

```text
# vikingbot

You are VikingBot, an AI assistant for searching and reading from the OpenViking context database.
You have access to tools that allow you to:
- Read, search, grep, and glob OpenViking files

IMPORTANT: Reply directly with your text response. Be accurate and concise.
```

检索用户提示 `P_ret(q)` 由 benchmark runner 拼接得到。在普通 FinanceBench bot 配置下，没有额外限定目录时，其形式为：

```text
Answer this question as briefly as possible. Use only the information available in the database.
Do not use any external source.
Always use OpenViking tools first. Search first, then read the results to answer.
Always search in viking://resources/ path.

Question: {q}
```

如果实验显式给定允许访问的目录集合 `U_allowed`，则第四句之后的范围约束会替换为：

```text
Always search ONLY within the following directories:
- {u_1}
- {u_2}
...
Do not access any other URI.

Efficiency tip:
- Prefer openviking_search in the allowed directory. This is usually more efficient than layer-by-layer grep.

Question: {q}
```

需要注意的是，`openviking_search`、`openviking_multi_read`、`openviking_grep` 等工具的名称、描述和参数 schema 并不是作为自然语言 prompt 写入 `M_0`，而是作为 LLM tool calling 接口的 `tools` 参数传入模型。因此论文正文可以把 `M_0` 写成“system instruction + retrieval instruction + user question”，把工具集合 `A` 单独写成可调用动作集合。

## Algorithm 1: AgentLoop 多轮检索

**Require:** 用户问题 `q`，最大轮数 `T`，固定工具集合 `A`，语言模型 `L`
**Ensure:** 最终答案 `a`

```text
Algorithm 1: Multi-round Retrieval in AgentLoop
01: Procedure AgentLoopRetrieve(q, T, A, L)
02:     M <- BuildInitialContext(q)
03:     t <- 0
04:     while t < T do
05:         t <- t + 1
06:         R_t <- L(M, A)
07:         if R_t does not request tool calls then
08:             return TextAnswer(R_t)
09:         g_t <- ParseRetrievalAction(R_t)
10:         o_t <- ExecuteRetrievalAction(A, g_t)
11:         M <- AppendObservation(M, g_t, o_t)
12:         M <- AppendReflectionPrompt(M)
13:     return FallbackAnswer(t, T)
14: end Procedure
```

### Algorithm 1 逐行解释

**第 01 行**定义一次问题回答的 agentic retrieval 入口。输入包括问题 `q`、最大轮数 `T`、固定工具集合 `A` 和语言模型 `L`。在贯穿例子中，`q` 是关于 Company X 2022 年 capital expenditure 的问题，`A` 包含语义搜索、批量读取和 grep 工具。

**第 02 行**构造初始上下文 `M`。在 eval benchmark 中，`M` 由系统提示 `P_sys` 和检索用户提示 `P_ret(q)` 组成；后者明确要求模型只使用数据库信息、优先使用 OpenViking 工具、先搜索再读取。

**第 03 行**初始化轮次计数器 `t`。它表示 agent 已经做过多少次“观察上下文并决定下一步”的动作。例子中，`t=0` 表示还没有搜索，也还没有读取任何财报片段。

**第 04 行**进入多轮循环。只要 `t<T`，agent 就可以继续检索或读取；在本文参照配置中，`T=15`。这保证了模型可以先粗检索，再按需读取和精确定位，而不是被限制在单次检索结果中。

**第 05 行**进入新一轮决策并递增轮数。例子中，第 1 轮可能用于搜索 “Company X capital expenditure 2022”，第 2 轮可能用于读取搜索命中的财报 URI，第 3 轮可能用于 grep “capital expenditures”。

**第 06 行**调用语言模型 `L` 进行本轮决策。模型看到的不是整库文档，而是当前上下文 `M` 以及可调用工具 `A`。如果 `M` 中还没有证据，模型通常会选择搜索；如果 `M` 中已有候选 URI，模型可能选择读取；如果 `M` 中已有长文档，模型可能选择 grep 精确定位。

**第 07 行**判断模型是否还需要工具。如果 `R_t` 不包含工具调用，说明模型认为当前上下文已经足够支持回答。例子中，当上下文已经包含 “capital expenditures were $X million in fiscal 2022” 这样的原文证据时，模型就可以停止检索。

**第 08 行**返回最终答案。例子中，答案会根据已读取或 grep 命中的证据给出 Company X 在 2022 财年的资本开支数值。论文伪代码中直接用 `return` 表达正常终止。

**第 09 行**如果模型还需要证据，则把模型响应解析为本轮检索动作 `g_t`。这里用单数 action 表达主流程，是因为实际回答过程中模型通常每轮选择一个主要动作，例如先搜索，再读取，再 grep。

**第 10 行**执行检索动作并得到观察结果 `o_t`。例子中，如果 `g_t` 是搜索，则 `o_t` 是候选 URI 和摘要；如果 `g_t` 是读取，则 `o_t` 是指定 URI 的正文；如果 `g_t` 是 grep，则 `o_t` 是命中的行。

**第 11 行**把检索动作和观察结果写回上下文 `M`。在例子中，搜索返回的候选 URI、读取返回的财报正文、grep 返回的命中行，都会被追加到 `M`，成为下一轮模型决策的证据。

**第 12 行**追加反思提示，使下一轮模型重新评估当前证据是否充分。例子中，如果搜索结果只给出相关财报摘要，模型会继续读取；如果读取结果很长但未直接定位数值，模型会继续 grep；如果 grep 已找到目标数值，模型会回答。

**第 13 行**处理达到最大轮数仍未返回答案的情况。该兜底分支避免 agent 在找不到充分证据时无限搜索。例子中，如果 15 轮内仍未找到 Company X 的目标数值，系统会返回超限或无答案提示。

**第 14 行**表示过程结束。这个算法的核心状态是上下文 `M`：每轮的搜索、读取和 grep 结果都进入 `M`，下一轮决策直接建立在这些证据之上。

## Algorithm 2: 标准 OpenViking 语义检索

**Require:** 查询 `q`，目标 URI `u`，返回数量 `k`，OpenViking 客户端 `V`  
**Ensure:** 格式化检索结果 `Y`，候选文档集合 `D`

```text
Algorithm 2: Standard OpenViking Semantic Search
01: Procedure OpenVikingSearch(q, u, k, V)
02:     B <- GetSearchBackend(V)
03:     k <- LoadDefaultSearchLimitOrFallback(k)
04:     X <- B.Search(query = q, target_uri = u, limit = 3k)
05:     if X is empty then
06:         return "No results found", empty list
07:     D <- NormalizeSearchResults(X)
08:     if D is empty then
09:         return Stringify(X), empty list
10:     D <- RemoveAbstractAndOverviewDocuments(D)
11:     if D is empty then
12:         return "No L2 content results found", empty list
13:     D <- TopK(SortByScoreDescending(D), k)
14:     Y <- FormatStandardSearchResults(D)
15:     return Y, D
16: end Procedure
```

### Algorithm 2 逐行解释

**第 01 行**定义标准 OpenViking 语义检索过程。该过程对应普通 bot 配置下的 `openviking_search`，不包含 reference/relation 扩展逻辑。

**第 02 行**选择实际搜索后端 `B`。该后端负责在指定 URI 范围内执行语义检索，并返回与查询相关的候选资源。

**第 03 行**确定检索返回数量 `k`。在 FinanceBench 普通 bot 配置中，`search_limit` 为 10；因此一次标准搜索最终保留 10 个候选文档。

**第 04 行**执行语义检索。这里实际请求 `3k` 个候选，而不是直接请求 `k` 个，因为后续会过滤掉 L0/L1 摘要层文档，提前多取一些结果能提高过滤后仍保留足够 L2 内容文档的概率。

**第 05 行**判断后端是否没有返回任何结果。若原始结果为空，说明当前查询在指定 URI 范围内没有召回候选。

**第 06 行**处理完全无结果的情况。算法直接返回无结果提示和空候选集合，LLM 下一轮可以据此换查询、换范围，或承认没有找到证据。

**第 07 行**将原始搜索结果 `X` 归一化为候选文档集合 `D`。这一步把不同格式的后端返回统一成可排序、可过滤、可展示的文档列表。

**第 08 行**判断归一化后是否仍没有候选文档。这个分支用于处理后端返回结构不符合预期、或者资源字段为空的情况。

**第 09 行**在归一化失败时返回原始结果字符串。这样做可以保留后端响应信息，避免直接丢弃可能对调试或模型判断有用的内容。

**第 10 行**过滤摘要层文档。VikingRAG 文档层级中，`.abstract.md` 和 `.overview.md` 更像索引摘要，普通问答需要优先使用 L2 内容文档作为证据，因此这里将这些摘要层结果移除。

**第 11 行**判断过滤后是否没有 L2 内容文档。如果所有召回结果都是摘要层文档，则本次搜索无法提供可直接作答的正文证据。

**第 12 行**处理 L2 文档为空的情况。算法返回明确提示，使下一轮 LLM 可以调整检索策略，而不是误以为已有可用正文证据。

**第 13 行**对候选文档按相关性分数降序排序，并截断到前 `k` 个。这个步骤把后端召回结果压缩为模型可阅读的候选列表，避免把过多低相关文档放入上下文。

**第 14 行**将候选集合格式化为标准搜索结果文本 `Y`。文本通常包含 rank、URI、score 和摘要，供 LLM 判断下一步是否读取原文。

**第 15 行**返回格式化结果和候选文档集合。`Y` 是写回 LLM 上下文的工具观察，`D` 是算法说明中的结构化候选集合。

**第 16 行**表示标准搜索过程结束。

## Algorithm 3: 并发批量读取 OpenViking 文档

**Require:** URI 列表 `U`，OpenViking 客户端 `V`，并发上限 `m`  
**Ensure:** 多文档完整内容 `Y`

```text
Algorithm 3: Concurrent Multi-Read
01: Procedure MultiRead(U, V, m)
02:     if U is empty then
03:         return "Error: No URIs provided"
04:     Q <- empty list
05:     for each uri in U do
06:         Q <- Q union {ConcurrentRead(V, uri, m)}
07:     end for
08:     Z <- AwaitAll(Q)
09:     Y <- Header(|U|)
10:     for each z in Z do
11:         Y <- Y union DelimitAndAppend(z.uri, z.content, z.success)
12:     end for
13:     return Y
14: end Procedure
```

### Algorithm 3 逐行解释

**第 01 行**定义批量读取过程。该过程对应 `openviking_multi_read`，用于一次性读取多个候选 URI 的完整内容，是从“看摘要”进入“看原文证据”的关键步骤。

**第 02 行**检查待读取 URI 集合 `U` 是否为空。空集合通常意味着模型没有正确从搜索结果中选择候选文档，或者前一轮搜索没有返回可读 URI。

**第 03 行**在 URI 为空时直接返回错误提示。该错误会作为工具观察写回上下文，下一轮 LLM 可以重新搜索或修正读取参数。

**第 04 行**初始化并发任务集合 `Q`。系统会限制同时读取的文档数量，避免一次读取过多文件造成资源压力。

**第 05 行**遍历每个待读取 URI。每个 URI 对应一个 OpenViking 文档资源，通常来自上一轮 `openviking_search` 返回的候选结果。

**第 06 行**为当前 URI 创建并发读取任务，并加入任务集合 `Q`。这里的 `ConcurrentRead` 表示受并发上限控制的异步读取操作。

**第 07 行**表示 URI 遍历结束。此时所有读取任务都已加入 `Q`，但还不一定完成。

**第 08 行**等待所有读取任务完成，并得到结果集合 `Z`。并发读取可以降低多文档读取的总延迟，尤其适合模型一次选择多个候选 URI 的情况。

**第 09 行**生成批量读取结果的头部。该头部告诉 LLM 本次读取了多少个资源，帮助模型理解后续文本由多个文档片段组成。

**第 10 行**遍历每个读取结果 `z`。结果中包含 URI、读取是否成功、正文内容或错误信息。

**第 11 行**将单个读取结果追加到输出文本 `Y` 中。每个文档都会带有开始和结束边界；成功时写入完整内容，失败时写入错误信息，这样 LLM 能区分不同文档的证据边界。

**第 12 行**表示所有读取结果都已完成格式化。此时 `Y` 包含多个文档的完整内容或错误信息。

**第 13 行**返回批量读取结果。该结果会被 agentloop 作为工具观察写入上下文，下一轮 LLM 可以直接基于原文内容回答或继续 grep。

**第 14 行**表示批量读取过程结束。

## Algorithm 4: 并发正则证据检索

**Require:** 搜索 URI `u`，正则模式集合 `P_g`，大小写开关 `c`，OpenViking 客户端 `V`，并发上限 `m`  
**Ensure:** 按 URI 合并的命中行 `Y`

```text
Algorithm 4: Concurrent Pattern-based Evidence Search
01: Procedure OpenVikingGrep(u, P_g, c, V, m)
02:     P_g <- NormalizePatternList(P_g)
03:     Q <- empty list
04:     for each pattern p in P_g do
05:         Q <- Q union {ConcurrentGrep(V, u, p, c, m)}
06:     end for
07:     Z <- AwaitAll(Q)
08:     G <- empty map from uri to matched lines
09:     n <- 0
10:     for each (p, matches) in Z do
11:         n <- n + |matches|
12:         for each match in matches do
13:             G[match.uri] <- G[match.uri] union {(match.line, match.content, p)}
14:         end for
15:     end for
16:     if G is empty then
17:         return "No matches found"
18:     Y <- FormatMergedMatches(G, n, |P_g|)
19:     return Y
20: end Procedure
```

### Algorithm 4 逐行解释

**第 01 行**定义正则证据检索过程。该过程对应 `openviking_grep`，适用于模型已经知道大致文档范围，但需要定位精确数字、短语、表格字段或原文表述的场景。

**第 02 行**规范化正则模式集合 `P_g`。如果模型只传入一个 pattern，系统会将其视为单元素列表；如果传入多个 pattern，则后续可以并发执行。

**第 03 行**初始化 grep 任务集合 `Q`。系统同样会限制并发数，避免同时执行过多正则搜索。

**第 04 行**遍历每个正则模式 `p`。多个 pattern 可以用于同时查找不同关键词、数值格式或同义表述。

**第 05 行**为当前 pattern 创建 grep 任务，并加入并发任务集合。每个任务会在目标 URI 范围内查找匹配行。

**第 06 行**表示所有 grep 任务已创建完成。此时任务集合中包含每个 pattern 对应的检索任务。

**第 07 行**等待所有 grep 任务完成，并得到结果集合 `Z`。并发执行使多个 pattern 的搜索可以同时进行。

**第 08 行**初始化按 URI 分组的结果映射 `G`。这样做是为了把不同 pattern 命中的行合并到同一个文档下，便于 LLM 按文档阅读证据。

**第 09 行**初始化总命中数 `n`。该数值用于后续结果头部，告诉 LLM 本次 grep 总共找到了多少条匹配。

**第 10 行**遍历每个 pattern 的命中结果。`matches` 是当前 pattern 找到的一组命中记录。

**第 11 行**累加当前 pattern 的命中数量。这样可以得到所有 pattern 的总命中规模。

**第 12 行**遍历当前 pattern 的每一条命中记录。每条记录通常包含 URI、行号和命中文本。

**第 13 行**将命中记录合并到 URI 对应的结果组中。该操作保留行号、内容和触发该命中的 pattern，使最终输出既能定位原文位置，也能说明命中原因。

**第 14 行**表示当前 pattern 的所有命中记录已经合并完成。

**第 15 行**表示所有 pattern 的结果都已经合并完成。此时 `G` 是按 URI 组织的证据集合。

**第 16 行**检查是否没有任何命中。如果 `G` 为空，说明指定 URI 范围内没有找到符合 pattern 的文本。

**第 17 行**返回无命中提示。这个提示会写回上下文，使下一轮 LLM 可以改用其他关键词、扩大搜索范围或回到语义检索。

**第 18 行**将合并后的命中结果格式化为文本。结果会按 URI 分组，并对每个 URI 下的命中行按行号排序，便于模型阅读和引用。

**第 19 行**返回格式化后的 grep 结果。该结果作为工具观察进入下一轮上下文，帮助模型基于精确证据生成答案。

**第 20 行**表示正则检索过程结束。

## 多轮检索如何展开

普通 bot 检索通常按以下模式展开，但具体顺序由 LLM 根据上下文动态决定。

1. 首轮 LLM 接收用户问题和工具列表，通常先调用 `openviking_search(q)` 获取候选文档。
2. `openviking_search` 返回候选 URI、分数和摘要，LLM 根据摘要判断哪些 URI 值得读取。
3. 若候选摘要不足以直接回答，LLM 在下一轮调用 `openviking_multi_read(U)` 读取一个或多个候选文档完整内容。
4. 若问题需要定位精确短语、数字或表述，LLM 可调用 `openviking_grep(u, P_g)` 在指定 URI 范围内做正则检索。
5. 每次工具调用结果都会被追加到 `M` 中，下一轮 LLM 基于新证据继续决策。
6. 当 LLM 判断证据充分时，它停止调用工具，直接输出最终答案。

## 终止条件与失败兜底

1. LLM 不再返回工具调用：`AgentLoop._run_agent_loop` 将当前文本响应作为最终答案，并退出循环。
2. 达到最大轮数：如果达到 `T` 轮仍没有最终答案，返回超限提示。
3. 搜索无结果：`openviking_search` 返回 `No results found for query: ...`。
4. L2 过滤后无结果：`openviking_search` 返回 `No L2 content results found for query: ...`。
5. 批量读取 URI 为空：`openviking_multi_read` 返回 `Error: No URIs provided.`。
6. grep 无命中：`openviking_grep` 返回 `No matches found ...`。
7. 工具异常：工具返回错误字符串，agentloop 会把错误也作为观察结果写回上下文，让下一轮 LLM 判断是否改用其他检索方式。

## 关键设计点

1. 多轮检索由 LLM tool calling 驱动，不是写死的检索轮数。
2. 主循环按“选择一个检索动作、执行该动作、写回观察结果”的方式展开；并发只作为具体工具内部的执行优化。
3. 标准搜索阶段先请求 `3k` 个结果，再过滤 L0/L1 文档并保留 top-`k` 个 L2 内容文档。
4. `openviking_multi_read` 是从摘要候选进入证据验证的关键步骤。
5. `openviking_grep` 用于精确定位关键词、数值、短语或表格附近文本。
6. 示例配置关闭建边，且不启用关系扩展，因此本文算法只覆盖普通检索路径。
