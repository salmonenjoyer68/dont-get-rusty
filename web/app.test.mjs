// Regression tests for the Web Demo's workspace browser, driven through the
// real `app.js` with a minimal DOM and `fetch` stub. No dependencies:
//
//   node --test web/app.test.mjs
//
// The seam is what a user can see: after navigating, the `#path` label and the
// `#files` list must describe the same directory, even when responses arrive
// out of order.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const APP_SOURCE = readFileSync(new URL('./app.js', import.meta.url), 'utf8');

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.className = '';
    this.hidden = false;
    this.open = false;
    this.value = '';
    this.disabled = false;
    this.type = '';
    this.title = '';
    this.dataset = {};
    this.listeners = new Map();
    this._textContent = '';
  }

  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
  }

  get textContent() {
    if (this.children.length) {
      return this.children.map((child) => child.textContent).join('');
    }
    return this._textContent;
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = children;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  showModal() {
    this.open = true;
  }

  close() {
    this.open = false;
  }

  focus() {}

  querySelectorAll(selector) {
    return findByClass(this, selector.slice(1));
  }
}

function findByClass(root, className, matches = []) {
  for (const child of root.children) {
    if (child.className.split(/\s+/).includes(className)) matches.push(child);
    findByClass(child, className, matches);
  }
  return matches;
}

const IDS = [
  'transcript',
  'status',
  'run',
  'task',
  'decision',
  'decision-question',
  'file-view',
  'path',
  'files',
  'audit',
  'model',
  'composer',
  'allow',
  'deny',
  'audit-refresh',
  'workspace-up',
  'decision-error',
];

let eventSources = [];

class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.onopen = null;
    this.onmessage = null;
    eventSources.push(this);
  }
}

const flush = async () => {
  for (let i = 0; i < 5; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
};

const dir = (name) => ({ name, kind: 'dir', size: 0 });
const okResponse = (body) => ({
  ok: true,
  status: 200,
  json: async () => body,
  text: async () => '',
});
const errorResponse = (status) => ({
  ok: false,
  status,
  json: async () => ({}),
  text: async () => '',
});
const acceptedResponse = () => ({
  ok: true,
  status: 204,
  json: async () => ({}),
  text: async () => '',
});

function makeApp() {
  eventSources = [];
  const byId = new Map(IDS.map((id) => [id, new FakeElement('div')]));
  const elements = {
    document: {
      querySelector(selector) {
        const element = byId.get(selector.slice(1));
        if (!element) throw new Error(`missing element ${selector}`);
        return element;
      },
      querySelectorAll() {
        return [];
      },
      createElement(tagName) {
        return new FakeElement(tagName);
      },
    },
    EventSource: FakeEventSource,
  };

  const pending = [];
  const answers = [];
  elements.fetch = (url, options = {}) => {
    if (url === '/api/state') {
      return Promise.resolve(okResponse({ model: 'test', running: false }));
    }
    if (url === '/api/audit') return Promise.resolve(okResponse([]));
    if (url.startsWith('/api/decision/')) {
      return new Promise((resolve, reject) => answers.push({ url, options, resolve, reject }));
    }
    assert.match(url, /^\/api\/workspace\?path=/);
    const path = new URL(url, 'http://localhost').searchParams.get('path') || '';
    return new Promise((resolve) => pending.push({ path, resolve }));
  };

  vm.runInContext(APP_SOURCE, vm.createContext(elements), {
    filename: 'web/app.js',
  });

  const take = async (path) => {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      const index = pending.findIndex((request) => request.path === path);
      if (index >= 0) return pending.splice(index, 1)[0];
      await flush();
    }
    throw new Error(`no pending request for ${JSON.stringify(path)}`);
  };

  const files = () => byId.get('files');
  const labels = () => files().children.map((item) => item.textContent);
  const button = (label) => {
    for (const item of files().children) {
      if (item.textContent === label) return item.children[0];
    }
    throw new Error(`no button labelled ${JSON.stringify(label)}`);
  };
  const click = (element) => {
    for (const listener of element.listeners.get('click') || []) listener();
  };
  const clickById = (id) => click(byId.get(id));

  const takeAnswer = async () => {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      if (answers.length) return answers.shift();
      await flush();
    }
    throw new Error('no pending decision answer request');
  };

  const ready = async () => {
    const root = await take('');
    root.resolve(okResponse([]));
    await flush();
  };

  const emit = (event) => {
    const source = eventSources.at(-1);
    assert.ok(source, 'the app connected to the event stream');
    source.onmessage({ data: JSON.stringify(event) });
  };

  return {
    path: () => byId.get('path').textContent,
    labels,
    button,
    click,
    clickById,
    take,
    takeAnswer,
    ready,
    emit,
    dialogOpen: () => byId.get('decision').open,
    question: () => byId.get('decision-question').textContent,
    decisionError: () => byId.get('decision-error').textContent,
    decisionErrorHidden: () => byId.get('decision-error').hidden,
    lines: () => byId.get('transcript').children.map((line) => line.textContent),
    pendingAnswers: () => answers.length,
  };
}

test('a stale parent listing cannot append entries into the newer directory', async () => {
  const app = makeApp();

  const root = await app.take('');
  root.resolve(okResponse([dir('a')]));
  await flush();
  assert.equal(app.path(), '/');
  assert.deepEqual(app.labels(), ['a/']);

  app.click(app.button('a/'));
  const a = await app.take('a');
  a.resolve(okResponse([dir('b')]));
  await flush();
  assert.equal(app.path(), '/a');
  assert.deepEqual(app.labels(), ['b/']);

  app.click(app.button('b/'));
  const ab = await app.take('a/b');
  ab.resolve(okResponse([dir('c')]));
  await flush();
  assert.equal(app.path(), '/a/b');
  assert.deepEqual(app.labels(), ['c/']);

  // Two rapid parent clicks: `/a/b` -> `/a` -> `/`. The first response is
  // slower than the second, so it lands after the root listing is rendered.
  app.clickById('workspace-up');
  app.clickById('workspace-up');
  const stale = await app.take('a');
  const latest = await app.take('');

  latest.resolve(okResponse([dir('a')]));
  await flush();
  assert.equal(app.path(), '/');
  assert.deepEqual(app.labels(), ['a/']);

  stale.resolve(okResponse([dir('b')]));
  await flush();
  assert.equal(app.path(), '/');
  assert.deepEqual(
    app.labels(),
    ['a/'],
    'a stale response must not append into the current directory',
  );
});

test('a stale failed listing cannot replace the newer directory with an error', async () => {
  const app = makeApp();

  const root = await app.take('');
  root.resolve(okResponse([dir('a')]));
  await flush();

  app.click(app.button('a/'));
  const stale = await app.take('a');
  app.clickById('workspace-up');
  const latest = await app.take('');

  latest.resolve(okResponse([dir('a')]));
  await flush();
  assert.equal(app.path(), '/');
  assert.deepEqual(app.labels(), ['a/']);

  stale.resolve(errorResponse(500));
  await flush();
  assert.equal(app.path(), '/');
  assert.deepEqual(
    app.labels(),
    ['a/'],
    'a stale failure must not replace the current listing',
  );
});

const decisionAsked = {
  type: 'decision_asked',
  id: 'd1',
  question: 'Overwrite notes.txt?',
};

test('a failed answer reopens the dialog and keeps the Confirmation retryable', async () => {
  const app = makeApp();
  await app.ready();
  app.emit(decisionAsked);
  assert.equal(app.dialogOpen(), true);
  assert.equal(app.question(), 'Overwrite notes.txt?');

  app.clickById('allow');
  const failed = await app.takeAnswer();
  assert.equal(failed.url, '/api/decision/d1');
  assert.deepEqual(JSON.parse(failed.options.body), { allow: true });
  failed.resolve(errorResponse(500));
  await flush();

  assert.equal(app.dialogOpen(), true, 'the dialog reopens so the user can retry');
  assert.equal(app.decisionErrorHidden(), false, 'the dialog explains the failure');
  assert.match(app.decisionError(), /500/);

  app.clickById('deny');
  const retry = await app.takeAnswer();
  assert.equal(retry.url, '/api/decision/d1');
  assert.deepEqual(JSON.parse(retry.options.body), { allow: false });
  retry.resolve(acceptedResponse());
  await flush();

  assert.equal(app.dialogOpen(), false);
  assert.equal(app.decisionErrorHidden(), true);
  assert.equal(app.pendingAnswers(), 0);
});

test('a rejected answer request reopens the dialog for a retry', async () => {
  const app = makeApp();
  await app.ready();
  app.emit(decisionAsked);

  app.clickById('allow');
  const failed = await app.takeAnswer();
  failed.reject(new TypeError('fetch failed'));
  await flush();

  assert.equal(app.dialogOpen(), true, 'the decision survives a network failure');
  assert.equal(app.decisionErrorHidden(), false);
  assert.match(app.decisionError(), /reach/i);

  app.clickById('allow');
  const retry = await app.takeAnswer();
  retry.resolve(acceptedResponse());
  await flush();
  assert.equal(app.dialogOpen(), false);
  assert.equal(app.decisionErrorHidden(), true);
});

test('a 404 answer is terminal: the dialog closes and the Confirmation is not retried', async () => {
  const app = makeApp();
  await app.ready();
  app.emit(decisionAsked);

  app.clickById('allow');
  const request = await app.takeAnswer();
  request.resolve(errorResponse(404));
  await flush();

  assert.equal(app.dialogOpen(), false, 'a lost response must not leave a stuck dialog');
  assert.equal(app.decisionErrorHidden(), true);
  assert.ok(
    app.lines().some((line) => /no longer pending/i.test(line)),
    'the transcript says why the Confirmation went away',
  );

  app.clickById('allow');
  await flush();
  assert.equal(app.pendingAnswers(), 0, 'a terminal answer is not retried');
});

test('a successful answer clears the Confirmation and closes the dialog', async () => {
  const app = makeApp();
  await app.ready();
  app.emit(decisionAsked);

  app.clickById('allow');
  const request = await app.takeAnswer();
  request.resolve(acceptedResponse());
  await flush();

  assert.equal(app.dialogOpen(), false);
  assert.equal(app.decisionErrorHidden(), true);

  app.clickById('allow');
  await flush();
  assert.equal(app.pendingAnswers(), 0, 'an answered Confirmation is not sent twice');
});

test('decision_answered still resolves a dialog that a failed POST reopened', async () => {
  const app = makeApp();
  await app.ready();
  app.emit(decisionAsked);

  app.clickById('allow');
  const failed = await app.takeAnswer();
  failed.resolve(errorResponse(500));
  await flush();
  assert.equal(app.dialogOpen(), true);

  app.emit({ type: 'decision_answered', id: 'd1', allow: true, timed_out: false });
  await flush();

  assert.equal(app.dialogOpen(), false, 'the server event closes the dialog');
  assert.equal(app.decisionErrorHidden(), true);
  assert.ok(app.lines().some((line) => line.includes('Confirmation approved.')));

  app.clickById('allow');
  await flush();
  assert.equal(app.pendingAnswers(), 0, 'the resolved Confirmation is not retried');
});
