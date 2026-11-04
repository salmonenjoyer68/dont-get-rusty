// Web Demo page: renders the SSE stream from the Rust server.
// No framework, no build step. Every server field is rendered with
// `textContent`, so model output can never become markup.

const $ = (selector) => document.querySelector(selector);
const transcript = $('#transcript');
const statusEl = $('#status');
const runButton = $('#run');
const taskInput = $('#task');
const decisionDialog = $('#decision');
const decisionQuestion = $('#decision-question');
const fileView = $('#file-view');

let running = false;
let currentDecision = null;
let pendingCalls = [];
let workspacePath = '';

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

function setStatus() {
  statusEl.textContent = running ? 'running' : 'idle';
  statusEl.className = `status ${running ? 'running' : 'idle'}`;
  runButton.disabled = running;
  runButton.textContent = running ? 'Running…' : 'Run Task';
}

function appendLine(className, text) {
  const line = el('div', `event ${className}`, text);
  transcript.append(line);
  return line;
}

function formatArguments(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return raw;
  }
  if (parsed === null || typeof parsed !== 'object') return raw;
  const parts = [];
  if (typeof parsed.path === 'string') parts.push(`path: ${parsed.path}`);
  if (typeof parsed.stdin === 'string') parts.push(`stdin:\n${parsed.stdin}`);
  if (typeof parsed.code === 'string') parts.push(parsed.code);
  if (typeof parsed.content === 'string') parts.push(parsed.content);
  return parts.length ? parts.join('\n\n') : JSON.stringify(parsed, null, 2);
}

function addCall(event) {
  const card = el('article', 'call');
  const head = el('div', 'call-head');
  const badge = el('span', 'badge', 'running');
  head.append(el('span', 'tool', event.tool), badge);

  const args = el('pre', 'args');
  args.textContent = formatArguments(event.arguments);

  const details = el('details', 'observation-wrap');
  const observation = el('pre', 'observation');
  details.append(el('summary', '', 'observation'), observation);

  card.append(head, args, details);
  transcript.append(card);
  pendingCalls.push({ tool: event.tool, badge, observation, details });
}

function addObservation(event) {
  const index = pendingCalls.findIndex((call) => call.tool === event.tool);
  const call = index >= 0 ? pendingCalls.splice(index, 1)[0] : null;

  const target = call ? call.observation : el('pre', 'observation');
  target.textContent = event.content;
  if (!call) {
    transcript.append(target);
    return;
  }

  const failed = event.timed_out || (event.exit_code !== null && event.exit_code !== 0);
  call.details.open = failed;
  if (event.timed_out) {
    call.badge.textContent = 'timeout';
    call.badge.className = 'badge warn';
  } else if (event.exit_code === null) {
    call.badge.textContent = 'host';
    call.badge.className = 'badge muted';
  } else {
    call.badge.textContent = `exit ${event.exit_code}`;
    call.badge.className = `badge ${event.exit_code === 0 ? 'ok' : 'error'}`;
  }
}

function askDecision(event) {
  currentDecision = event.id;
  decisionQuestion.textContent = event.question;
  appendLine('decision', 'Waiting for a Confirmation…');
  if (!decisionDialog.open) decisionDialog.showModal();
}

function resolveDecision(event) {
  if (currentDecision === event.id) {
    currentDecision = null;
    if (decisionDialog.open) decisionDialog.close();
  }
  if (event.timed_out) {
    appendLine('decision', 'Confirmation timed out; answered deny.');
  } else {
    appendLine('decision', event.allow ? 'Confirmation approved.' : 'Confirmation denied.');
  }
}

function finish(event) {
  running = false;
  setStatus();
  if (event.answer) {
    // The final Step's assistant text is the same answer; drop the echo.
    const echoes = transcript.querySelectorAll('.event.assistant');
    const last = echoes[echoes.length - 1];
    if (last && last.textContent === event.answer) last.remove();
    appendLine('answer', event.answer);
  }
  if (event.error) appendLine('errorbox', event.error);
  appendLine('meta', `${event.steps} Step(s) · ${(event.duration_ms / 1000).toFixed(1)} s`);
  refreshWorkspace(workspacePath);
  refreshAudit();
}

function handleEvent(event) {
  switch (event.type) {
    case 'task_started':
      transcript.replaceChildren();
      pendingCalls = [];
      running = true;
      setStatus();
      fileView.hidden = true;
      appendLine('task', `Task: ${event.task}`);
      break;
    case 'step_started':
      appendLine('step', `Step ${event.step + 1}`);
      break;
    case 'assistant_text':
      appendLine('assistant', event.text);
      break;
    case 'call_started':
      addCall(event);
      break;
    case 'observation':
      addObservation(event);
      break;
    case 'decision_asked':
      askDecision(event);
      break;
    case 'decision_answered':
      resolveDecision(event);
      break;
    case 'task_finished':
      finish(event);
      break;
    default:
      break;
  }
}

async function answerDecision(allow) {
  const id = currentDecision;
  if (!id) return;
  currentDecision = null;
  if (decisionDialog.open) decisionDialog.close();
  await fetch(`/api/decision/${encodeURIComponent(id)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ allow }),
  });
}

async function submitTask(task) {
  if (running) return;
  const response = await fetch('/api/task', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ task }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    appendLine('errorbox', body.error || `could not start the Task (${response.status})`);
  }
}

async function refreshWorkspace(path) {
  workspacePath = path;
  $('#path').textContent = `/${path}`;
  const list = $('#files');
  list.replaceChildren();
  fileView.hidden = true;

  const response = await fetch(`/api/workspace?path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    list.append(el('li', 'hint', 'cannot list this directory'));
    return;
  }
  const entries = await response.json();
  for (const entry of entries) {
    const child = path ? `${path}/${entry.name}` : entry.name;
    const button = el(
      'button',
      `file ${entry.kind}`,
      entry.kind === 'file' ? `${entry.name}  ·  ${entry.size} B` : `${entry.name}/`,
    );
    button.type = 'button';
    button.addEventListener('click', () => {
      if (entry.kind === 'dir') refreshWorkspace(child);
      else openFile(child);
    });
    const item = el('li');
    item.append(button);
    list.append(item);
  }
  if (!entries.length) list.append(el('li', 'hint', '(empty)'));
}

async function openFile(path) {
  const response = await fetch(`/api/workspace/file?path=${encodeURIComponent(path)}`);
  if (!response.ok) return;
  fileView.textContent = `--- ${path} ---\n\n${await response.text()}`;
  fileView.hidden = false;
}

async function refreshAudit() {
  const box = $('#audit');
  box.replaceChildren();
  const response = await fetch('/api/audit');
  if (!response.ok) {
    box.append(el('p', 'hint', 'cannot read the Audit Log'));
    return;
  }
  const events = await response.json();
  for (const event of events.slice().reverse()) {
    const bits = [event.ts, `step ${event.step}`, event.tool, event.decision];
    if (event.exit_code !== null && event.exit_code !== undefined) {
      bits.push(`exit ${event.exit_code}`);
    }
    if (event.error) bits.push(event.error);
    const line = el('div', `audit-line ${event.decision === 'deny' ? 'deny' : ''}`);
    line.textContent = bits.join(' · ');
    line.title = JSON.stringify(event, null, 2);
    box.append(line);
  }
  if (!events.length) box.append(el('p', 'hint', 'no events yet'));
}

function connect() {
  const source = new EventSource('/api/events?after=0');
  let opened = false;
  source.onopen = () => {
    // A reconnect replays the retained transcript from the start, so clear
    // the DOM instead of appending duplicates.
    if (opened) {
      transcript.replaceChildren();
      pendingCalls = [];
    }
    opened = true;
  };
  source.onmessage = (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    handleEvent(event);
  };
}

$('#composer').addEventListener('submit', (event) => {
  event.preventDefault();
  const task = taskInput.value.trim();
  if (task) submitTask(task);
});

for (const preset of document.querySelectorAll('[data-task]')) {
  preset.addEventListener('click', () => {
    taskInput.value = preset.dataset.task;
    taskInput.focus();
  });
}

$('#allow').addEventListener('click', () => answerDecision(true));
$('#deny').addEventListener('click', () => answerDecision(false));
$('#audit-refresh').addEventListener('click', refreshAudit);
$('#workspace-up').addEventListener('click', () => {
  const parent = workspacePath.split('/').slice(0, -1).join('/');
  refreshWorkspace(parent);
});
decisionDialog.addEventListener('cancel', (event) => {
  event.preventDefault();
  answerDecision(false);
});

async function load() {
  const response = await fetch('/api/state');
  if (response.ok) {
    const state = await response.json();
    $('#model').textContent = state.model;
    running = state.running;
  }
  setStatus();
  await Promise.all([refreshWorkspace(''), refreshAudit()]);
  connect();
}

load();
