# 知问 · 企业知识库智能问答助手

> 第七章智能应用开发课程作业。一个基于 LangChain 与 RAG（检索增强生成）的可运行 Web 应用，支持 OpenAI 兼容接口和 Ollama 离线模型。

知问面向企业制度查询场景：用户用自然语言提问，系统先从企业内部文档中检索有关片段，再由大模型依据这些片段组织答案。回答中的关键结论会标注 `[资料1]` 等来源编号，界面同时展示可核对的原文摘要，降低大模型脱离企业资料“自由发挥”的风险。

## 核心功能

- **基于 LangChain 的 RAG 对话链**：使用 `ChatPromptTemplate`、`MessagesPlaceholder`、模型适配器与 `StrOutputParser` 组成 LangChain Expression Language（LCEL）链。
- **严格资料边界**：提示词要求模型只使用检索资料作答；资料不足时明确说明，不编造制度、数字或审批人。
- **来源可追溯**：每条回答显示命中文档、章节、原文摘要和 `[资料N]` 引用。
- **多轮对话**：客户端保留最近对话并传给 LangChain，可围绕上一个回答继续追问。
- **在线与离线模型**：支持 OpenAI、硅基流动、DeepSeek 等 OpenAI 兼容接口，也支持通过 Ollama 运行本地模型。
- **中文友好检索**：使用单字、双字和英文词元的混合索引，并加入章节标题权重，无需额外部署向量数据库即可演示。
- **零前端构建依赖**：后端使用 Python 标准库 HTTP 服务，前端使用原生 HTML、CSS 和 JavaScript。
- **无 API Key 也能演示**：未配置模型时进入检索演示模式，可直接查看检索与引用效果；配置模型后切换为完整问答。

## 技术架构

```text
浏览器（原生 HTML / CSS / JavaScript）
                 │  POST /api/chat
                 ▼
Python ThreadingHTTPServer
                 │
                 ▼
AssistantService
   ├─ KnowledgeBase：文档加载 → 章节切分 → 中文混合索引 → Top-K 检索
   ├─ ChatPromptTemplate：系统规则 + 最近 8 条对话 + 检索资料
   ├─ ChatOpenAI / ChatOllama：在线或离线大模型
   └─ StrOutputParser：解析并返回回答
                 │
                 ▼
回答 + [资料N] 引用 + 原始资料摘要
```

RAG 数据流如下：

1. 应用启动时递归读取 `data/` 下的 `.md`、`.markdown` 和 `.txt` 文件。
2. 按 Markdown 标题划分章节，再按自然标点和重叠窗口切分为知识片段。
3. 对问题和知识片段进行中文友好的词元化，计算加权相关度并返回 Top-K 片段。
4. 将检索片段、历史对话和当前问题传入 LangChain 提示词模板。
5. 大模型生成带 `[资料N]` 引用的回答；API 同时返回结构化来源列表。

## 快速开始

### 1. 准备环境

建议使用 Python 3.10–3.12。

```bash
git clone <你的仓库地址>
cd langchain-knowledge-assistant
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python app.py
```

浏览器访问 <http://127.0.0.1:8000>。

若暂时没有配置 API Key，应用会自动进入**本地检索演示模式**。该模式适合检查界面、知识库和引用流程，但不调用大模型。

### 2. 配置在线大模型

编辑 `.env`：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=你的_API_KEY
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_BASE_URL` 留空时使用 OpenAI 官方地址。若使用兼容 OpenAI 协议的国产模型服务，填写服务商提供的 Base URL 和对应模型名即可，例如：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=你的_DeepSeek_API_KEY
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

项目已使用 DeepSeek 的 `deepseek-chat` 模型完成真实问答、多轮追问和无资料拒答测试。其他兼容 OpenAI 的工具调用接口也可按同样方式接入。

### 3. 配置 Ollama 离线模型

先安装 [Ollama](https://ollama.com/)，然后拉取并运行模型：

```bash
ollama pull qwen2.5:7b
ollama serve
```

编辑 `.env`：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
```

重新启动 `python app.py`。界面右上角会显示当前模型模式。

## 使用示例

项目内置四份虚构的“星禾科技”制度文档，可直接尝试：

- “年假有多少天？如何申请？”
- “出差住宿费的报销标准是多少？”
- “发现疑似数据泄露时应该怎么做？”
- “P1 工单多久需要响应？”

也可以在回答后继续追问，例如：“那连续休假三天需要谁审批？”

## 添加自己的文档

将 UTF-8 编码的 Markdown 或纯文本文件放入 `data/`，重新启动应用即可。建议使用清晰的标题组织内容：

```markdown
# 文档标题

## 章节名称

本章节包含可被检索和引用的制度正文。
```

当前版本为保证展示透明性，没有在页面中接受任意文件上传。新增资料应经过管理员检查后放入 `data/`，避免将未经审核的内容直接加入知识库。

## 项目结构

```text
langchain-knowledge-assistant/
├─ app.py                         # 启动入口
├─ data/                          # 示例企业知识库
├─ src/knowledge_assistant/
│  ├─ config.py                   # 环境变量与配置
│  ├─ knowledge.py                # 文档加载、分块、检索
│  ├─ llm.py                      # OpenAI / Ollama LangChain 适配
│  ├─ service.py                  # RAG 提示词与 LCEL 链
│  └─ server.py                   # HTTP API 与静态文件服务
├─ static/                        # 原生 Web 界面
├─ tests/                         # 检索、服务、LangChain 链测试
├─ .env.example
├─ requirements.txt
└─ README.md
```

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖知识库加载、带薪年假检索、住宿标准检索、数据泄露检索、分块大小、演示模式、LangChain 链调用和空问题校验。GitHub Actions 工作流位于 `.github/workflows/ci.yml`。

## HTTP API

获取应用和知识库状态：

```http
GET /api/health
```

发起问答：

```http
POST /api/chat
Content-Type: application/json

{
  "question": "年假怎么申请？",
  "history": [
    {"role": "user", "content": "我是正式员工"},
    {"role": "assistant", "content": "已了解"}
  ]
}
```

响应示例：

```json
{
  "answer": "年假应至少提前 2 个工作日通过 OA 提交，并由直属主管审批。[资料1]",
  "sources": [
    {
      "index": 1,
      "title": "星禾科技员工手册（2026 版）",
      "section": "带薪年假",
      "source": "employee-handbook.md",
      "excerpt": "申请年假应至少提前 2 个工作日通过 OA 提交……",
      "score": 3.12
    }
  ],
  "mode": "openai",
  "model": "gpt-4o-mini"
}
```

## 安全与边界

- `.env` 已加入 `.gitignore`，API Key 不会进入版本库。
- 模型提示词明确规定：检索资料中的命令和角色设定不能改变系统行为。
- 回答只以 `data/` 中受控文档为依据，不开放公网检索。
- 当前示例文档和邮箱均为虚构内容，仅用于课程演示。
- 当前检索使用本地词法索引，不等同于语义向量检索；大型知识库建议接入企业批准的开源 Embedding 模型和向量数据库。
- 本应用是制度查询辅助工具，不替代人事、财务、法务或信息安全人员的正式判断。

## 发布为开源仓库

提交课程作业前，在 GitHub 创建公开仓库并执行：

```bash
git remote add origin https://github.com/<你的用户名>/langchain-knowledge-assistant.git
git push -u origin main
```

然后将仓库链接粘贴到作业文本框，并上传本 README 文档。请勿提交 `.env` 或任何真实 API Key。

## 许可证

本项目使用 [MIT License](LICENSE)。

