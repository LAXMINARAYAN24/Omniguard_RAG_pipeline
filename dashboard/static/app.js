/**
 * app.js — OmniGuard-RAG Interactive Studio Frontend Application
 *
 * Provides reactive UI logic for:
 * 1. Live Chat with Grounded Local LLM & Multi-Defense execution
 * 2. Real-time 4-Ring Telemetry Drawer inspection
 * 3. Controlled Attack Playground (Clean, Standard, PIDP, Collusion, Stealth, Silent, Custom)
 * 4. Multi-System Side-by-Side Comparison Modal
 * 5. Custom Poison Document Injector & Local LLM Provider Configuration
 */

// Application State
const state = {
  topics: [],
  activeTopicId: null,
  activeSystem: "omniguard",
  activeAttack: "collusion_stealth",
  kPoison: 3,
  hasRepetitiveSuffix: false,
  persistTrust: true,
  temperature: 0.2,
  messages: [],
  currentTelemetry: null,
  lastQueryPayload: null,
  llmConfig: {
    provider: "builtin",
    url: "http://localhost:11434",
    model: "llama3:8b"
  }
};

// DOM Element Selectors
const elements = {
  sidebar: document.getElementById("sidebar"),
  toggleSidebarBtn: document.getElementById("toggleSidebarBtn"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  themeIcon: document.getElementById("themeIcon"),
  themeLabel: document.getElementById("themeLabel"),
  newChatBtn: document.getElementById("newChatBtn"),
  resetCorpusBtn: document.getElementById("resetCorpusBtn"),
  openSettingsBtn: document.getElementById("openSettingsBtn"),
  openInjectModalBtn: document.getElementById("openInjectModalBtn"),
  headerCompareBtn: document.getElementById("headerCompareBtn"),

  // Status & Badges
  serverIndicator: document.getElementById("serverIndicator"),
  activeLLMBadge: document.getElementById("activeLLMBadge"),
  corpusDocCount: document.getElementById("corpusDocCount"),
  customPoisonCount: document.getElementById("customPoisonCount"),
  topicCountBadge: document.getElementById("topicCountBadge"),
  topicsContainer: document.getElementById("topicsContainer"),

  // Chat Area
  chatContainer: document.getElementById("chatContainer"),
  welcomeScreen: document.getElementById("welcomeScreen"),
  messagesList: document.getElementById("messagesList"),
  sessionTitle: document.getElementById("sessionTitle"),

  // Input & Attack Controls
  systemSelect: document.getElementById("systemSelect"),
  attackTypeSelect: document.getElementById("attackTypeSelect"),
  kPoisonSlider: document.getElementById("kPoisonSlider"),
  kPoisonValue: document.getElementById("kPoisonValue"),
  poisonSliderGroup: document.getElementById("poisonSliderGroup"),
  suffixCheckbox: document.getElementById("suffixCheckbox"),
  trustCheckbox: document.getElementById("trustCheckbox"),
  queryInput: document.getElementById("queryInput"),
  sendBtn: document.getElementById("sendBtn"),

  // Telemetry Drawer
  telemetryDrawer: document.getElementById("telemetryDrawer"),
  closeDrawerBtn: document.getElementById("closeDrawerBtn"),
  drawerRouteBadge: document.getElementById("drawerRouteBadge"),
  drawerContent: document.getElementById("drawerContent"),

  // Modals
  compareModal: document.getElementById("compareModal"),
  closeCompareModalBtn: document.getElementById("closeCompareModalBtn"),
  compareModalBody: document.getElementById("compareModalBody"),
  compareQuerySubtitle: document.getElementById("compareQuerySubtitle"),

  injectModal: document.getElementById("injectModal"),
  closeInjectModalBtn: document.getElementById("closeInjectModalBtn"),
  cancelInjectBtn: document.getElementById("cancelInjectBtn"),
  submitInjectBtn: document.getElementById("submitInjectBtn"),
  injectTopicSelect: document.getElementById("injectTopicSelect"),
  injectTargetAnswer: document.getElementById("injectTargetAnswer"),
  injectText: document.getElementById("injectText"),

  settingsModal: document.getElementById("settingsModal"),
  closeSettingsModalBtn: document.getElementById("closeSettingsModalBtn"),
  saveSettingsBtn: document.getElementById("saveSettingsBtn"),
  testConnectionBtn: document.getElementById("testConnectionBtn"),
  llmProviderSelect: document.getElementById("llmProviderSelect"),
  endpointUrlGroup: document.getElementById("endpointUrlGroup"),
  llmEndpointUrl: document.getElementById("llmEndpointUrl"),
  modelSelectGroup: document.getElementById("modelSelectGroup"),
  llmModelInput: document.getElementById("llmModelInput"),
  llmTempSlider: document.getElementById("llmTempSlider"),
  tempValue: document.getElementById("tempValue"),
  llmTestStatusBox: document.getElementById("llmTestStatusBox")
};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadInitialData();
});

function initEventListeners() {
  // Theme Toggle
  elements.themeToggleBtn.addEventListener("click", toggleTheme);

  // Sidebar Toggle on Mobile
  elements.toggleSidebarBtn?.addEventListener("click", () => {
    elements.sidebar.classList.toggle("open");
  });

  // Reset / New Chat
  elements.newChatBtn.addEventListener("click", resetChatSession);
  elements.resetCorpusBtn.addEventListener("click", handleCorpusReset);

  // Input Auto-resize & Keydown
  elements.queryInput.addEventListener("input", autoResizeTextarea);
  elements.queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });
  elements.sendBtn.addEventListener("click", handleSendMessage);

  // Attack Options
  elements.attackTypeSelect.addEventListener("change", (e) => {
    state.activeAttack = e.target.value;
    if (state.activeAttack === "clean") {
      elements.poisonSliderGroup.style.display = "none";
    } else {
      elements.poisonSliderGroup.style.display = "flex";
    }
  });

  elements.kPoisonSlider.addEventListener("input", (e) => {
    state.kPoison = parseInt(e.target.value, 10);
    elements.kPoisonValue.textContent = state.kPoison;
  });

  elements.suffixCheckbox.addEventListener("change", (e) => {
    state.hasRepetitiveSuffix = e.target.checked;
  });

  elements.trustCheckbox.addEventListener("change", (e) => {
    state.persistTrust = e.target.checked;
  });

  elements.systemSelect.addEventListener("change", (e) => {
    state.activeSystem = e.target.value;
  });

  // Feature Cards in Welcome Screen
  document.querySelectorAll(".feature-card").forEach((card) => {
    card.addEventListener("click", () => {
      const topicId = parseInt(card.getAttribute("data-topic-id"), 10);
      const promptText = card.querySelector(".card-prompt").textContent.replace(/"/g, "");
      selectTopic(topicId);
      elements.queryInput.value = promptText;
      autoResizeTextarea();
      elements.queryInput.focus();
    });
  });

  // Drawer Controls
  elements.closeDrawerBtn.addEventListener("click", () => {
    elements.telemetryDrawer.classList.remove("open");
  });

  // Header Compare Button
  elements.headerCompareBtn.addEventListener("click", () => {
    const text = elements.queryInput.value.trim() || (state.lastQueryPayload?.query?.raw_text);
    if (text) {
      openCompareModal(text);
    } else {
      openCompareModal("What is the key result regarding chlorophyll?");
    }
  });

  // Modals Close handlers
  elements.closeCompareModalBtn.addEventListener("click", () => elements.compareModal.style.display = "none");
  elements.closeInjectModalBtn.addEventListener("click", () => elements.injectModal.style.display = "none");
  elements.cancelInjectBtn.addEventListener("click", () => elements.injectModal.style.display = "none");
  elements.openInjectModalBtn.addEventListener("click", openInjectModal);
  elements.submitInjectBtn.addEventListener("click", handleSubmitCustomPoison);

  elements.openSettingsBtn.addEventListener("click", openSettingsModal);
  elements.closeSettingsModalBtn.addEventListener("click", () => elements.settingsModal.style.display = "none");
  elements.saveSettingsBtn.addEventListener("click", handleSaveLLMSettings);
  elements.testConnectionBtn.addEventListener("click", handleTestLLMConnection);

  elements.llmProviderSelect.addEventListener("change", handleLLMProviderChange);
  elements.llmTempSlider.addEventListener("input", (e) => {
    state.temperature = parseFloat(e.target.value);
    elements.tempValue.textContent = state.temperature.toFixed(2);
  });
}

// --- API Calls & Data Loaders ---

async function loadInitialData() {
  try {
    const [statusRes, topicsRes] = await Promise.all([
      fetch("/api/status").then(r => r.json()),
      fetch("/api/topics").then(r => r.json())
    ]);

    // Update Status
    updateSystemStatus(statusRes);

    // Populate Topics
    state.topics = topicsRes.topics || [];
    renderTopicsList(state.topics);
    populateInjectTopics(state.topics);

  } catch (err) {
    console.error("Error loading initial data:", err);
    elements.serverIndicator.className = "status-indicator";
  }
}

function updateSystemStatus(status) {
  if (status.status === "healthy") {
    elements.serverIndicator.className = "status-indicator online";
    elements.corpusDocCount.textContent = `${status.corpus.clean_docs_count} Clean`;
    elements.customPoisonCount.textContent = `${status.corpus.custom_poison_count} Active`;

    const llm = status.llm;
    if (llm.active_provider === "builtin") {
      elements.activeLLMBadge.textContent = "Built-in Synthesizer";
    } else if (llm.active_provider === "ollama") {
      elements.activeLLMBadge.textContent = `Ollama (${llm.active_model})`;
    } else {
      elements.activeLLMBadge.textContent = `OpenAI-Compat (${llm.active_model})`;
    }
  }
}

function renderTopicsList(topics) {
  elements.topicCountBadge.textContent = topics.length;
  elements.topicsContainer.innerHTML = "";

  topics.forEach((t) => {
    const item = document.createElement("div");
    item.className = `topic-item ${state.activeTopicId === t.topic_id ? "active" : ""}`;
    item.innerHTML = `
      <div class="topic-name">${escapeHTML(t.display_name)}</div>
      <div class="topic-meta">Ans: ${escapeHTML(String(t.answer))}</div>
    `;
    item.addEventListener("click", () => {
      selectTopic(t.topic_id);
      const sample = t.sample_queries[0] || `Explain ${t.name}`;
      elements.queryInput.value = sample;
      autoResizeTextarea();
      elements.queryInput.focus();
    });
    elements.topicsContainer.appendChild(item);
  });
}

function populateInjectTopics(topics) {
  elements.injectTopicSelect.innerHTML = "";
  topics.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.topic_id;
    opt.textContent = `${t.topic_id}: ${t.display_name}`;
    elements.injectTopicSelect.appendChild(opt);
  });
}

function selectTopic(topicId) {
  state.activeTopicId = topicId;
  const topic = state.topics.find(t => t.topic_id === topicId);
  if (topic) {
    elements.sessionTitle.textContent = `${topic.display_name} — Defense Analysis`;
  }
  document.querySelectorAll(".topic-item").forEach((el, idx) => {
    el.classList.toggle("active", idx === topicId);
  });
}

// --- Chat Actions ---

async function handleSendMessage() {
  const text = elements.queryInput.value.trim();
  if (!text) return;

  // Clear welcome screen on first message
  if (elements.welcomeScreen.style.display !== "none") {
    elements.welcomeScreen.style.display = "none";
    elements.messagesList.style.display = "flex";
  }

  // Suffix formulation if checkbox active
  let suffix = null;
  if (state.hasRepetitiveSuffix) {
    suffix = "syn ack syn ack syn ack syn ack syn ack syn ack syn ack syn ack syn ack syn ack";
  }

  const payload = {
    query: text,
    topic_id: state.activeTopicId,
    attack_type: state.activeAttack,
    k_poison: state.kPoison,
    adversarial_suffix: suffix,
    persist_trust: state.persistTrust,
    system: state.activeSystem,
    temperature: state.temperature
  };

  // Append User Message to UI
  appendUserMessage(text, payload);

  // Clear input
  elements.queryInput.value = "";
  autoResizeTextarea();
  elements.sendBtn.disabled = true;

  // Append Assistant Loading Skeleton
  const loadingBubble = appendAssistantLoading();

  try {
    const t0 = performance.now();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);

    if (data.status === "error" || data.error) {
      loadingBubble.innerHTML = `<div class="text-danger" style="padding: 8px;">⚠️ ${escapeHTML(data.error || "Server processing error")}</div>`;
      return;
    }

    // Replace Loading Skeleton with Rich Response
    renderAssistantResponse(loadingBubble, data, elapsed);

    // Store as latest telemetry
    state.lastQueryPayload = data;
    state.currentTelemetry = data.telemetry;

    // Refresh Corpus Stats
    refreshStatusBadge();

  } catch (err) {
    console.error("Chat request failed:", err);
    loadingBubble.innerHTML = `<div class="text-danger" style="padding: 8px;">⚠️ Error connecting to RAG defense server: ${escapeHTML(err.message)}</div>`;
  } finally {
    elements.sendBtn.disabled = false;
    scrollChatToBottom();
  }
}

function appendUserMessage(text, payload) {
  const row = document.createElement("div");
  row.className = "message-row user-row";

  const attackBadge = formatAttackBadge(payload.attack_type, payload.k_poison, payload.adversarial_suffix);

  row.innerHTML = `
    <div class="message-bubble-wrapper">
      <div class="message-meta-header">
        ${attackBadge}
        <span>You</span>
      </div>
      <div class="message-bubble">
        ${escapeHTML(text)}
      </div>
    </div>
    <div class="message-avatar">👤</div>
  `;
  elements.messagesList.appendChild(row);
  scrollChatToBottom();
}

function appendAssistantLoading() {
  const row = document.createElement("div");
  row.className = "message-row assistant-row";
  row.innerHTML = `
    <div class="message-avatar">🛡️</div>
    <div class="message-bubble-wrapper">
      <div class="message-meta-header">
        <span class="badge badge-accent">Processing Defense...</span>
      </div>
      <div class="message-bubble">
        <div class="loading-spinner-wrap" style="padding: 12px 0;">
          <div class="spinner"></div>
          <span style="font-size: 12px;">Evaluating 4-Ring Defense & Local LLM Synthesis...</span>
        </div>
      </div>
    </div>
  `;
  elements.messagesList.appendChild(row);
  scrollChatToBottom();
  return row;
}

function renderAssistantResponse(rowElement, data, clientLatency) {
  const defense = data.defense;
  const llm = data.llm_generation || {};

  const isSuccess = defense.is_correct;
  const isPoisoned = defense.is_attack_success;

  const outcomeBadge = isSuccess
    ? `<span class="badge badge-success">✓ Ground Truth Verified</span>`
    : (isPoisoned ? `<span class="badge badge-danger">⚠️ Attacker Target Injected</span>` : `<span class="badge badge-warning">⚠️ Refused / Inconclusive</span>`);

  const routeBadge = defense.route === "fast"
    ? `<span class="badge badge-accent">⚡ Fast Path (1x Call)</span>`
    : `<span class="badge badge-info">🛡️ Deep GWCC (${defense.calls}x Calls)</span>`;

  const groundingBadge = llm.abstention_triggered
    ? `<span class="badge badge-warning">⚠️ Abstention (IDK Policy)</span>`
    : (llm.grounding_status === "VERIFIED"
        ? `<span class="badge badge-success">✓ CoV Grounded (${Math.round((llm.confidence_score || 1.0) * 100)}% Conf)</span>`
        : `<span class="badge badge-ghost">Grounding: ${escapeHTML(llm.grounding_status || "Standard")}</span>`);

  const parsedMarkdown = renderSimpleMarkdown(llm.text || "");

  rowElement.innerHTML = `
    <div class="message-avatar">🛡️</div>
    <div class="message-bubble-wrapper">
      <div class="message-meta-header">
        <span class="badge badge-accent">${escapeHTML(defense.system_name)}</span>
        ${outcomeBadge}
        ${routeBadge}
        ${groundingBadge}
        <span>• ${defense.pipeline_latency_ms}ms pipeline | ${llm.latency_ms || 0}ms LLM</span>
      </div>
      <div class="message-bubble">
        ${parsedMarkdown}
      </div>
      <div class="message-actions">
        ${data.telemetry ? `<button class="btn btn-secondary btn-xs inspect-tel-btn">🔬 Inspect 4-Ring Defense</button>` : ''}
        <button class="btn btn-secondary btn-xs compare-sys-btn">⚖️ Compare 6 Systems</button>
        <button class="btn btn-ghost btn-xs copy-btn">📋 Copy</button>
      </div>
    </div>
  `;

  // Attach button events
  const inspectBtn = rowElement.querySelector(".inspect-tel-btn");
  if (inspectBtn) {
    inspectBtn.addEventListener("click", () => {
      openTelemetryDrawer(data);
    });
  }

  const compareBtn = rowElement.querySelector(".compare-sys-btn");
  if (compareBtn) {
    compareBtn.addEventListener("click", () => {
      openCompareModal(data.query.raw_text);
    });
  }

  const copyBtn = rowElement.querySelector(".copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(llm.text);
      copyBtn.textContent = "✓ Copied!";
      setTimeout(() => copyBtn.textContent = "📋 Copy", 1500);
    });
  }
}

// --- 4-Ring Telemetry Drawer Rendering ---

function openTelemetryDrawer(data) {
  state.currentTelemetry = data.telemetry;
  const tel = data.telemetry;
  if (!tel) return;

  const routeName = data.defense.route.toUpperCase();
  elements.drawerRouteBadge.textContent = `${routeName} PATH (${data.defense.calls} Calls)`;
  elements.drawerRouteBadge.className = data.defense.route === "fast" ? "badge badge-accent" : "badge badge-info";

  let html = `
    <!-- Ring 0 Card -->
    <div class="tel-card">
      <div class="tel-card-header">
        <div class="tel-card-title">
          <span class="pill-dot ring0"></span>
          <strong>Ring 0: Query-Path Guard</strong>
        </div>
        <span class="badge ${tel.ring0.flagged ? 'badge-danger' : 'badge-success'}">
          ${tel.ring0.flagged ? '⚠️ Suffix Flagged & Stripped' : '✓ Query Clean'}
        </span>
      </div>
      <div class="tel-row">
        <span class="tel-label">Repetition Ratio:</span>
        <span class="tel-value">${tel.ring0.repetition_ratio} / ${tel.ring0.threshold}</span>
      </div>
      <div class="metric-bar-wrap">
        <div class="metric-bar-fill" style="width: ${Math.min(100, (tel.ring0.repetition_ratio / 1.0) * 100)}%; background-color: ${tel.ring0.flagged ? 'var(--danger)' : 'var(--success)'};"></div>
      </div>
      <div class="tel-row">
        <span class="tel-label">Action:</span>
        <span class="tel-value">${escapeHTML(tel.ring0.action_taken)}</span>
      </div>
      <div class="tel-row">
        <span class="tel-label">Sanitized Text:</span>
        <span class="tel-value"><code>"${escapeHTML(tel.ring0.sanitized_text)}"</code></span>
      </div>
    </div>

    <!-- Ring 1 Card -->
    <div class="tel-card">
      <div class="tel-card-header">
        <div class="tel-card-title">
          <span class="pill-dot ring1"></span>
          <strong>Ring 1: Spectral DRS Outlier Filter</strong>
        </div>
        <span class="badge ${tel.ring1.dropped_docs_count > 0 ? 'badge-warning' : 'badge-success'}">
          ${tel.ring1.dropped_docs_count} Poison Outliers Dropped
        </span>
      </div>
      <div class="tel-row">
        <span class="tel-label">Candidate Pool:</span>
        <span class="tel-value">${tel.ring1.total_candidate_docs} Total (${tel.ring1.kept_docs_count} Kept, ${tel.ring1.dropped_docs_count} Dropped)</span>
      </div>
      <div class="tel-row">
        <span class="tel-label">Spectral SVD Threshold:</span>
        <span class="tel-value">${tel.ring1.drs_threshold}</span>
      </div>
      ${tel.ring1.dropped_documents.length > 0 ? `
        <div class="tel-row" style="margin-top: 4px;">
          <span class="tel-label">Dropped Outlier Passages:</span>
        </div>
        <div class="doc-card-list">
          ${tel.ring1.dropped_documents.map(d => `
            <div class="doc-mini-card poison">
              <div class="doc-card-top">
                <span class="doc-id-badge text-danger">⚠️ ${escapeHTML(d.doc_id)}</span>
                <span class="badge badge-danger">Outlier Score: ${d.anomaly_score}</span>
              </div>
              <div class="doc-text-snippet">"${escapeHTML(d.snippet)}"</div>
            </div>
          `).join('')}
        </div>
      ` : ''}
    </div>

    <!-- Retrieved Top-5 Passages Table -->
    <div class="tel-card">
      <div class="tel-card-header">
        <div class="tel-card-title">
          <strong>Top-5 Retrieved Passages</strong>
        </div>
        <span class="badge badge-accent">k=5 Passages</span>
      </div>
      <div class="doc-card-list">
        ${tel.retrieval.map(d => `
          <div class="doc-mini-card ${d.is_poison ? 'poison' : 'clean'}">
            <div class="doc-card-top">
              <span class="doc-id-badge">${d.rank}. ${escapeHTML(d.doc_id)} ${d.is_poison ? '⚠️ [POISON]' : '✓ [CLEAN]'}</span>
              <span>Sim: <strong>${d.cosine_similarity}</strong> | Trust: <strong>${d.trust_score}</strong></span>
            </div>
            <div class="doc-text-snippet">"${escapeHTML(d.text_snippet)}"</div>
            <div class="tel-row" style="font-size: 11px; margin-top: 4px;">
              <span class="tel-label">Asserted Answer:</span>
              <span class="tel-value"><code>${escapeHTML(d.claim_answer)}</code></span>
            </div>
          </div>
        `).join('')}
      </div>
    </div>

    <!-- Ring 2 Card -->
    <div class="tel-card">
      <div class="tel-card-header">
        <div class="tel-card-title">
          <span class="pill-dot ring2"></span>
          <strong>Ring 2: Risk-Aware Router</strong>
        </div>
        <span class="badge ${tel.ring2.route_decision === 'deep' ? 'badge-warning' : 'badge-success'}">
          Route: ${escapeHTML(tel.ring2.route_decision.toUpperCase())}
        </span>
      </div>
      <div class="tel-row">
        <span class="tel-label">Embedding Cohesion (Sim):</span>
        <span class="tel-value">${tel.ring2.embedding_cohesion} (Threshold: ${tel.ring2.cohesion_threshold})</span>
      </div>
      <div class="metric-bar-wrap">
        <div class="metric-bar-fill" style="width: ${(tel.ring2.embedding_cohesion / 1.0) * 100}%; background-color: ${tel.ring2.embedding_cohesion >= tel.ring2.cohesion_threshold ? 'var(--success)' : 'var(--warning)'};"></div>
      </div>
      <div class="tel-row">
        <span class="tel-label">Answer Contention:</span>
        <span class="tel-value">${(tel.ring2.answer_contention * 100).toFixed(1)}% (Floor: ${(tel.ring2.contention_threshold * 100).toFixed(0)}%)</span>
      </div>
      <div class="metric-bar-wrap">
        <div class="metric-bar-fill" style="width: ${(tel.ring2.answer_contention / 1.0) * 100}%; background-color: ${tel.ring2.answer_contention < tel.ring2.contention_threshold ? 'var(--success)' : 'var(--danger)'};"></div>
      </div>
      <div class="tel-row">
        <span class="tel-label">Decision Reason:</span>
        <span class="tel-value" style="font-size: 11px;">${escapeHTML(tel.ring2.escalation_reason)}</span>
      </div>
    </div>

    <!-- Ring 3 Card -->
    <div class="tel-card">
      <div class="tel-card-header">
        <div class="tel-card-title">
          <span class="pill-dot ring3"></span>
          <strong>Ring 3: GWCC Consensus & Causal Sensitivity</strong>
        </div>
        <span class="badge ${tel.ring3.invoked ? 'badge-info' : 'badge-ghost'}">
          ${tel.ring3.invoked ? 'Active Counterfactual Pass' : 'Bypassed (Fast Path)'}
        </span>
      </div>
      ${tel.ring3.invoked ? `
        <div class="tel-row">
          <span class="tel-label">Leave-One-Out Flips:</span>
          <span class="tel-value">${tel.ring3.leave_one_out_flips.length > 0 ? escapeHTML(tel.ring3.leave_one_out_flips.join(', ')) : 'None'}</span>
        </div>
        <div class="tel-row">
          <span class="tel-label">Implicated Poison Clique:</span>
          <span class="tel-value text-danger">${tel.ring3.excluded_doc_ids.length > 0 ? escapeHTML(tel.ring3.excluded_doc_ids.join(', ')) : 'None'}</span>
        </div>
        <div class="tel-row">
          <span class="tel-label">Consensus Verified Answer:</span>
          <span class="tel-value text-success"><strong>${escapeHTML(tel.ring3.consensus_answer)}</strong></span>
        </div>
        ${tel.ring3.sensitivity_summary ? `
          <div class="tel-row" style="margin-top: 4px;">
            <span class="tel-label">Causal Sensitivity Diagnosis:</span>
            <span class="tel-value" style="font-size: 11px; color: var(--accent-primary);">${escapeHTML(tel.ring3.sensitivity_summary)}</span>
          </div>
        ` : ''}
        ${tel.ring3.counterfactual_loo_scores && Object.keys(tel.ring3.counterfactual_loo_scores).length > 0 ? `
          <div class="tel-row" style="margin-top: 4px;">
            <span class="tel-label">LOO Sensitivity Scores S(d_i):</span>
          </div>
          <div class="sensitivity-grid">
            ${Object.entries(tel.ring3.counterfactual_loo_scores).map(([docId, score]) => `
              <div class="sensitivity-cell ${score > 0 ? 'flip' : 'stable'}">
                <code>${escapeHTML(docId)}</code>
                <span>${score > 0 ? '⚠️ Flip (1.0)' : '✓ 0.0'}</span>
              </div>
            `).join('')}
          </div>
        ` : ''}
        ${tel.ring3.counterfactual_lgo_scores && Object.keys(tel.ring3.counterfactual_lgo_scores).length > 0 ? `
          <div class="tel-row" style="margin-top: 4px;">
            <span class="tel-label">LGO Collusion Sensitivity S(d_i, d_j):</span>
          </div>
          <div class="sensitivity-grid">
            ${Object.entries(tel.ring3.counterfactual_lgo_scores).map(([pair, score]) => `
              <div class="sensitivity-cell ${score > 0 ? 'flip' : 'stable'}">
                <code>${escapeHTML(pair)}</code>
                <span>${score > 0 ? '⚠️ Clique Flip (1.0)' : '✓ 0.0'}</span>
              </div>
            `).join('')}
          </div>
        ` : ''}
      ` : `
        <div class="tel-row">
          <span class="tel-label">Status:</span>
          <span class="tel-value">Fast weighted majority applied without counterfactual overhead.</span>
        </div>
      `}
    </div>

    <!-- Research-Backed Factual Grounding & CoV Card -->
    ${data.llm_generation ? `
      <div class="tel-card">
        <div class="tel-card-header">
          <div class="tel-card-title">
            <strong>Factual Grounding & Chain-of-Verification (CoV)</strong>
          </div>
          <span class="badge ${data.llm_generation.abstention_triggered ? 'badge-warning' : (data.llm_generation.grounding_status === 'VERIFIED' ? 'badge-success' : 'badge-ghost')}">
            ${escapeHTML(data.llm_generation.grounding_status || 'STANDARD')}
          </span>
        </div>
        <div class="tel-row">
          <span class="tel-label">Grounding Protocol:</span>
          <span class="tel-value">${escapeHTML((data.llm_generation.grounding_mode || 'chain_of_verification').toUpperCase())}</span>
        </div>
        <div class="tel-row">
          <span class="tel-label">Factual Confidence Score:</span>
          <span class="tel-value text-success">${Math.round((data.llm_generation.confidence_score || 1.0) * 100)}%</span>
        </div>
        ${data.llm_generation.citations && data.llm_generation.citations.length > 0 ? `
          <div class="tel-row" style="margin-top: 4px;">
            <span class="tel-label">Mandated Context Citations:</span>
          </div>
          <div class="citations-wrap">
            ${data.llm_generation.citations.map(c => `
              <span class="citation-tag">[${escapeHTML(c)}]</span>
            `).join('')}
          </div>
        ` : ''}
        ${data.llm_generation.cov_questions && data.llm_generation.cov_questions.length > 0 ? `
          <div class="tel-row" style="margin-top: 4px;">
            <span class="tel-label">Premise-by-Premise Verification:</span>
          </div>
          <div class="cov-question-list">
            ${data.llm_generation.cov_questions.map(q => `
              <div class="cov-question-item ${q.status === 'SUPPORTED' ? 'supported' : 'unverified'}">
                <div class="cov-q-header">
                  <span>${escapeHTML(q.question_id.toUpperCase())}</span>
                  <span class="badge ${q.status === 'SUPPORTED' ? 'badge-success' : 'badge-danger'}">${escapeHTML(q.status)}</span>
                </div>
                <div class="cov-q-text">${escapeHTML(q.question)}</div>
                <div class="cov-q-footer">
                  <span>Docs: ${escapeHTML((q.supporting_docs || []).join(', ') || 'N/A')}</span>
                  <span>${escapeHTML(q.note || '')}</span>
                </div>
              </div>
            `).join('')}
          </div>
        ` : ''}
      </div>
    ` : ''}

    <!-- Dynamic Trust Updates -->
    ${tel.trust_store_updates && tel.trust_store_updates.length > 0 ? `
      <div class="tel-card">
        <div class="tel-card-header">
          <div class="tel-card-title">
            <strong>Dynamic Trust Store Updates</strong>
          </div>
          <span class="badge badge-accent">${tel.trust_store_updates.length} Updates</span>
        </div>
        <div class="doc-card-list">
          ${tel.trust_store_updates.map(u => `
            <div class="tel-row" style="font-size: 11px;">
              <span><code>${escapeHTML(u.doc_id)}</code>:</span>
              <span class="${u.change === 'penalized' ? 'text-danger' : 'text-success'}">
                ${u.previous} ➔ <strong>${u.new}</strong> (${escapeHTML(u.change)})
              </span>
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}
  `;

  elements.drawerContent.innerHTML = html;
  elements.telemetryDrawer.classList.add("open");
}

// --- Side-by-Side Comparison Modal ---

async function openCompareModal(queryText) {
  elements.compareQuerySubtitle.textContent = `Evaluating "${queryText}" across all 6 defense systems under ${state.activeAttack.toUpperCase()} attack...`;
  elements.compareModalBody.innerHTML = `
    <div class="loading-spinner-wrap">
      <div class="spinner"></div>
      <p>Simulating and evaluating 6 systems...</p>
    </div>
  `;
  elements.compareModal.style.display = "flex";

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryText,
        topic_id: state.activeTopicId,
        attack_type: state.activeAttack,
        k_poison: state.kPoison,
        adversarial_suffix: state.hasRepetitiveSuffix ? "syn ack syn ack syn ack syn ack syn ack" : null
      })
    });
    const data = await res.json();
    renderComparisonTable(data);
  } catch (err) {
    elements.compareModalBody.innerHTML = `<div class="text-danger">Failed to run comparison: ${escapeHTML(err.message)}</div>`;
  }
}

function renderComparisonTable(data) {
  const systems = data.systems_comparison || [];
  const groundTruth = data.query.ground_truth_answer;

  let rowsHtml = systems.map((sys) => {
    const isOmni = sys.system_id === "omniguard";
    const statusBadge = sys.is_correct
      ? `<span class="badge badge-success">✓ Correct (${escapeHTML(sys.answer)})</span>`
      : (sys.is_attack_success ? `<span class="badge badge-danger">❌ Poisoned (${escapeHTML(sys.answer)})</span>` : `<span class="badge badge-warning">⚠️ Inconclusive</span>`);

    return `
      <tr class="${isOmni ? 'highlight-omni' : ''}">
        <td><strong>${escapeHTML(sys.name)}</strong></td>
        <td>${statusBadge}</td>
        <td><span class="badge ${sys.calls > 1 ? 'badge-info' : 'badge-accent'}">${sys.calls}x Calls</span></td>
        <td>${sys.latency_ms} ms</td>
        <td style="font-size: 11px; color: var(--text-secondary);">${escapeHTML(sys.defense_action)}</td>
      </tr>
    `;
  }).join('');

  elements.compareModalBody.innerHTML = `
    <div style="margin-bottom: 12px; font-size: 13px;">
      <strong>Ground Truth Answer:</strong> <code class="text-success">${escapeHTML(groundTruth)}</code> |
      <strong>Active Attack:</strong> <code>${escapeHTML(data.query.attack_type)} (k=${data.query.k_poison})</code>
    </div>
    <table class="compare-table">
      <thead>
        <tr>
          <th>System Architecture</th>
          <th>Outcome / Answer</th>
          <th>Compute Overhead</th>
          <th>Latency</th>
          <th>Defense Strategy</th>
        </tr>
      </thead>
      <tbody>
        ${rowsHtml}
      </tbody>
    </table>
  `;
}

// --- Custom Poison Injector ---

function openInjectModal() {
  elements.injectModal.style.display = "flex";
}

async function handleSubmitCustomPoison() {
  const topicId = parseInt(elements.injectTopicSelect.value, 10);
  const text = elements.injectText.value.trim();
  const targetAnswer = elements.injectTargetAnswer.value.trim() || "ATTACKER_TARGET";

  if (!text) {
    alert("Please enter poison document text.");
    return;
  }

  try {
    const res = await fetch("/api/inject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic_id: topicId, text: text, target_answer: targetAnswer })
    });
    const data = await res.json();
    if (data.status === "injected") {
      alert(`Adversarial Document '${data.doc_id}' successfully injected into live pool!`);
      elements.injectModal.style.display = "none";
      elements.injectText.value = "";
      refreshStatusBadge();
    }
  } catch (err) {
    alert(`Error injecting poison: ${err.message}`);
  }
}

// --- Local LLM Settings ---

function openSettingsModal() {
  elements.settingsModal.style.display = "flex";
  handleLLMProviderChange();
}

function handleLLMProviderChange() {
  const p = elements.llmProviderSelect.value;
  if (p === "builtin") {
    elements.endpointUrlGroup.style.display = "none";
    elements.modelSelectGroup.style.display = "none";
  } else if (p === "ollama") {
    elements.endpointUrlGroup.style.display = "flex";
    elements.modelSelectGroup.style.display = "flex";
    elements.llmEndpointUrl.value = "http://localhost:11434";
    elements.llmModelInput.value = "llama3:8b";
  } else {
    elements.endpointUrlGroup.style.display = "flex";
    elements.modelSelectGroup.style.display = "flex";
    elements.llmEndpointUrl.value = "http://localhost:1234/v1";
    elements.llmModelInput.value = "local-model";
  }
}

async function handleTestLLMConnection() {
  const provider = elements.llmProviderSelect.value;
  const url = elements.llmEndpointUrl.value.trim();
  elements.llmTestStatusBox.innerHTML = `<span class="status-msg">Probing ${escapeHTML(provider)} endpoint...</span>`;

  try {
    const res = await fetch("/api/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, url })
    });
    const data = await res.json();
    if (data.available) {
      const modelList = (data.models || []).slice(0, 5).map(m => escapeHTML(m)).join(', ');
      elements.llmTestStatusBox.innerHTML = `
        <span class="status-msg text-success">✓ ${escapeHTML(data.message)}</span>
        ${modelList.length > 0 ? `<div style="font-size: 11px; margin-top: 4px;">Models: ${modelList}</div>` : ''}
      `;
    } else {
      elements.llmTestStatusBox.innerHTML = `<span class="status-msg text-danger">✕ ${escapeHTML(data.message)}</span>`;
    }
  } catch (err) {
    elements.llmTestStatusBox.innerHTML = `<span class="status-msg text-danger">✕ Connection failed: ${escapeHTML(err.message)}</span>`;
  }
}

async function handleSaveLLMSettings() {
  const provider = elements.llmProviderSelect.value;
  const url = elements.llmEndpointUrl.value.trim();
  const model = elements.llmModelInput.value.trim();

  try {
    const res = await fetch("/api/llm/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, url, model })
    });
    const data = await res.json();
    if (data.status === "updated") {
      updateSystemStatus(await fetch("/api/status").then(r => r.json()));
      elements.settingsModal.style.display = "none";
    }
  } catch (err) {
    alert(`Failed to save settings: ${err.message}`);
  }
}

// --- Corpus Reset ---

async function handleCorpusReset() {
  if (confirm("Reset dynamic trust store scores to 1.0 and clear all custom-injected poisons?")) {
    try {
      await fetch("/api/corpus/reset", { method: "POST" });
      refreshStatusBadge();
      alert("Corpus and Trust Store successfully reset.");
    } catch (err) {
      alert(`Reset failed: ${err.message}`);
    }
  }
}

async function refreshStatusBadge() {
  const status = await fetch("/api/status").then(r => r.json());
  updateSystemStatus(status);
}

// --- UI Utilities ---

function resetChatSession() {
  elements.messagesList.innerHTML = "";
  elements.messagesList.style.display = "none";
  elements.welcomeScreen.style.display = "flex";
  elements.telemetryDrawer.classList.remove("open");
  state.activeTopicId = null;
  document.querySelectorAll(".topic-item").forEach(el => el.classList.remove("active"));
  elements.sessionTitle.textContent = "OmniGuard-RAG Playground";
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", next);
  elements.themeIcon.textContent = next === "dark" ? "☀️" : "🌙";
  elements.themeLabel.textContent = next === "dark" ? "Light Mode" : "Dark Mode";
}

function autoResizeTextarea() {
  elements.queryInput.style.height = "auto";
  elements.queryInput.style.height = `${Math.min(elements.queryInput.scrollHeight, 140)}px`;
}

function scrollChatToBottom() {
  elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

function formatAttackBadge(attackType, k, hasSuffix) {
  let label = attackType;
  let cls = "badge-danger";
  if (attackType === "clean") {
    label = "🛡️ Clean";
    cls = "badge-success";
  } else if (attackType === "standard") {
    label = `🎯 Standard (k=${k})`;
  } else if (attackType === "pidp") {
    label = `💉 PIDP + Suffix`;
  } else if (attackType.startsWith("collusion")) {
    label = `🥷 Collusion (k=${k})`;
  } else if (attackType === "silent") {
    label = `👻 Silent Subspace`;
  }
  return `<span class="badge ${cls}">${label}</span>`;
}

function renderSimpleMarkdown(text) {
  if (!text) return "";
  let html = escapeHTML(text);

  // Bold **text**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Headers ###
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');

  // Code inline `code`
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');

  // Bullet points
  html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

  // Blockquotes
  html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Paragraphs
  html = html.replace(/\n\n/g, '<br><br>');

  return html;
}

function escapeHTML(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
