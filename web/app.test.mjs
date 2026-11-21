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
];

class FakeEventSource {
  constructor(url) {
    this.url = url;
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

function makeApp() {
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
  elements.fetch = (url) => {
    if (url === '/api/state') {
      return Promise.resolve(okResponse({ model: 'test', running: false }));
    }
    if (url === '/api/audit') return Promise.resolve(okResponse([]));
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

  return {
    path: () => byId.get('path').textContent,
    labels,
    button,
    click,
    clickById,
    take,
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
