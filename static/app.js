const state = {
  messages: [],
  busy: false,
  mode: "demo",
};

const elements = {
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#questionInput"),
  send: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  documentList: document.querySelector("#documentList"),
  documentCount: document.querySelector("#documentCount"),
  chunkCount: document.querySelector("#chunkCount"),
  statusBadge: document.querySelector("#statusBadge"),
  statusText: document.querySelector("#statusText"),
  modelName: document.querySelector("#modelName"),
  clear: document.querySelector("#clearButton"),
  toast: document.querySelector("#toast"),
  suggestions: document.querySelector("#suggestions"),
};

let toastTimer;
let statusRetryTimer;
let statusRetryCount = 0;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `请求失败（${response.status}）`);
  }
  return payload;
}

async function loadStatus() {
  try {
    const status = await api("/api/health");
    statusRetryCount = 0;
    applyStatus(status);
  } catch (error) {
    elements.statusBadge.className = "status-badge";
    elements.statusText.textContent = "连接失败";
    elements.modelName.textContent = "服务未连接";
    showToast(error.message);
    if (statusRetryCount < 5) {
      statusRetryCount += 1;
      clearTimeout(statusRetryTimer);
      statusRetryTimer = setTimeout(loadStatus, 2000);
    }
  }
}

function applyStatus(status) {
  state.mode = status.mode;
  elements.documentCount.textContent = status.documents.length;
  elements.chunkCount.textContent = status.chunk_count;
  renderDocuments(status.documents);

  const isDemo = status.mode === "demo";
  elements.statusBadge.className = `status-badge ${isDemo ? "demo" : "ready"}`;
  elements.statusText.textContent = isDemo ? "演示模式" : "知识库已就绪";
  elements.modelName.textContent = `${providerLabel(status.mode)} · ${status.model}`;
  elements.modelName.parentElement.className = `model-chip ${isDemo ? "demo" : "ready"}`;
}

function providerLabel(mode) {
  return (
    {
      openai: "OpenAI",
      ollama: "Ollama",
      demo: "本地检索",
    }[mode] || mode
  );
}

function renderDocuments(documents) {
  elements.documentList.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "panel-note";
    empty.textContent = "知识库暂无文档。";
    elements.documentList.append(empty);
    return;
  }

  documents.forEach((documentInfo, index) => {
    const item = document.createElement("div");
    item.className = "document-item";
    item.style.animationDelay = `${index * 70}ms`;
    const filename = documentInfo.title.replace(/\.[^.]+$/, "");
    item.innerHTML = `
      <span class="document-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24">
          <path d="M6 3.5h8l4 4V21H6z"></path>
          <path d="M14 3.5V8h4M9 12h6M9 15.5h6"></path>
        </svg>
      </span>
      <span>
        <strong></strong>
        <small>${filename.endsWith(".md") ? "Markdown 文档" : "知识文档"}</small>
      </span>
      <span class="document-count">${document.chunk_count} 段</span>
    `;
    item.querySelector("strong").textContent = documentInfo.title;
    item.title = documentInfo.source;
    elements.documentList.append(item);
  });
}

function setBusy(busy) {
  state.busy = busy;
  elements.send.disabled = busy;
  elements.input.disabled = busy;
}

function resizeTextarea() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 130)}px`;
}

function addUserMessage(text) {
  removeWelcome();
  const article = document.createElement("article");
  article.className = "message user";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  article.append(body);
  elements.messages.append(article);
  scrollToBottom();
}

function addTypingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant loading-message";
  article.innerHTML = `
    <div class="message-avatar" aria-hidden="true">问</div>
    <div class="message-body">
      <div class="message-author">知问</div>
      <div class="typing" aria-label="正在检索并生成回答">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  elements.messages.append(article);
  scrollToBottom();
  return article;
}

function addAssistantMessage(content, sources = [], isError = false) {
  removeWelcome();
  const article = document.createElement("article");
  article.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "问";

  const body = document.createElement("div");
  body.className = "message-body";
  const author = document.createElement("div");
  author.className = "message-author";
  author.textContent = isError ? "系统提示" : "知问";
  body.append(author);

  const answer = document.createElement("div");
  answer.className = "answer-text";
  renderAnswer(answer, content);
  body.append(answer);

  if (sources.length) {
    body.append(createSources(sources));
  }

  article.append(avatar, body);
  elements.messages.append(article);
  scrollToBottom();
}

function renderAnswer(container, text) {
  let currentList = null;
  let currentListType = null;

  for (const rawLine of String(text).split("\n")) {
    const line = rawLine.trimEnd();
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);

    if (bullet || numbered) {
      const listType = bullet ? "ul" : "ol";
      if (!currentList || currentListType !== listType) {
        currentList = document.createElement(listType);
        currentListType = listType;
        container.append(currentList);
      }
      const item = document.createElement("li");
      appendInlineContent(item, (bullet || numbered)[1]);
      currentList.append(item);
      continue;
    }

    currentList = null;
    currentListType = null;
    if (!line.trim()) {
      continue;
    }
    const paragraph = document.createElement("p");
    appendInlineContent(paragraph, line);
    container.append(paragraph);
  }
}

function appendInlineContent(container, text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|`[^`]+`|\[资料\d+\])/g);
  for (const part of parts) {
    if (!part) continue;
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      container.append(strong);
    } else if (/^`[^`]+`$/.test(part)) {
      const code = document.createElement("code");
      code.textContent = part.slice(1, -1);
      container.append(code);
    } else if (/^\[资料\d+\]$/.test(part)) {
      const citation = document.createElement("span");
      citation.className = "citation";
      citation.textContent = part;
      container.append(citation);
    } else {
      container.append(document.createTextNode(part));
    }
  }
}

function createSources(sources) {
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `查看 ${sources.length} 条参考资料`;
  const list = document.createElement("div");
  list.className = "source-list";

  sources.forEach((source) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const top = document.createElement("div");
    top.className = "source-card-top";
    const title = document.createElement("strong");
    title.textContent = `${source.title} · ${source.section}`;
    title.title = title.textContent;
    const index = document.createElement("span");
    index.className = "source-index";
    index.textContent = `资料${source.index}`;
    top.append(title, index);
    const excerpt = document.createElement("p");
    excerpt.textContent = source.excerpt;
    card.append(top, excerpt);
    card.title = source.source;
    list.append(card);
  });

  details.append(summary, list);
  return details;
}

function removeWelcome() {
  elements.welcome?.remove();
  elements.welcome = null;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  });
}

async function ask(question) {
  const cleaned = question.trim();
  if (!cleaned || state.busy) return;

  const history = state.messages.slice(-8);
  setBusy(true);
  addUserMessage(cleaned);
  state.messages.push({ role: "user", content: cleaned });
  elements.input.value = "";
  resizeTextarea();
  const typing = addTypingMessage();

  try {
    const response = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ question: cleaned, history }),
    });
    if (elements.statusText.textContent === "连接失败") {
      await loadStatus();
    }
    typing.remove();
    addAssistantMessage(response.answer, response.sources);
    state.messages.push({ role: "assistant", content: response.answer });
  } catch (error) {
    typing.remove();
    addAssistantMessage(error.message, [], true);
    showToast(error.message);
  } finally {
    setBusy(false);
    elements.input.focus();
  }
}

function showToast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 3800);
}

function resetConversation() {
  if (!state.messages.length || state.busy) return;
  state.messages = [];
  elements.messages.replaceChildren();
  window.location.reload();
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(elements.input.value);
});

elements.input.addEventListener("input", resizeTextarea);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

elements.suggestions?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (button) ask(button.dataset.question);
});

elements.clear.addEventListener("click", resetConversation);

loadStatus();
resizeTextarea();

