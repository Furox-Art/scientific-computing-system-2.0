/**
 * Regression test for the npm publish visibility poll.
 *
 * The bug this guards against is a FALSE FAILURE: `npm publish` returned, the
 * version really was published, but the verification loop gave up too early and
 * the run reported a hard error (run 37120536500 on 5.2.5, which logged attempts
 * 1/6 through 6/6 and then failed, even though 5.2.5 is on the registry).
 *
 * Both directions are asserted, because fixing a false failure by deleting the
 * check would be worse than the bug:
 *
 *   1. 404-then-success -> must resolve and stop on first match.
 *      This is the direction that was broken.
 *   2. never appears    -> must still fail after the whole budget.
 *      A poll that cannot fail is not a safety check.
 *
 * Additional guards:
 *   - the budget must exceed the observed propagation window (~95s) and the
 *     registry's own `Cache-Control: max-age=300`, so a future edit cannot
 *     quietly shrink the loop back to the broken ~60s;
 *   - the backoff must stay bounded (no unbounded sleep);
 *   - each read must defeat caching, because a stale packument is one of the
 *     two root causes;
 *   - the subprocess CLI must exit 0 and non-zero in the two directions.
 *
 * Run via `npm test`.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

import { backoffSeconds, hasVersion, waitForVisibility } from "./wait-for-npm-visibility.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const scriptPath = join(here, "wait-for-npm-visibility.mjs");
// Spawned children need a real file:// URL: a bare Windows path is rejected by
// the ESM loader (ERR_UNSUPPORTED_ESM_URL_SCHEME).
const moduleUrl = pathToFileURL(scriptPath).href;

const PACKAGE = "scientific-computing-system-2.0";
const VERSION = "5.2.5";
const noSleep = async () => {};
const quiet = () => {};

const checks = [];

/** Register a check. Async bodies are awaited by `runChecks`. */
function check(name, fn) {
  checks.push({ name, fn });
}

async function runChecks() {
  let failed = 0;
  for (const { name, fn } of checks) {
    try {
      await fn();
      console.log(`ok - ${name}`);
    } catch (error) {
      failed += 1;
      console.error(`not ok - ${name}`);
      console.error(`    ${String(error.message).split("\n").join("\n    ")}`);
    }
  }
  console.log(`\n${checks.length - failed}/${checks.length} visibility checks passed`);
  return failed;
}

/** A fetch double that answers 404 for the first `misses` calls, then 200. */
function flakyFetch(misses, body) {
  const state = { calls: 0, urls: [], headers: [] };
  const impl = async (url, options = {}) => {
    state.calls += 1;
    state.urls.push(url);
    state.headers.push(options.headers ?? {});
    if (state.calls <= misses) {
      return { ok: false, status: 404, json: async () => ({ error: "Not found" }) };
    }
    return {
      ok: true,
      status: 200,
      json: async () =>
        body ?? { versions: { [VERSION]: { dist: { tarball: "https://example.invalid/x.tgz" } } } },
    };
  };
  return { impl, state };
}

/** Run the exported wait loop inside a child process, so its exit code is real. */
function runInChild(source) {
  const harness = `
    import { waitForVisibility } from ${JSON.stringify(moduleUrl)};
    ${source}
  `;
  return spawnSync(process.execPath, ["--input-type=module", "--eval", harness], { encoding: "utf8" });
}

check("a version missing from a well-formed packument is not visible", () => {
  assert.equal(hasVersion({ versions: { "1.0.0": {} } }, VERSION), false);
  assert.equal(hasVersion({ versions: { [VERSION]: {} } }, VERSION), true);
  assert.equal(hasVersion(null, VERSION), false);
  assert.equal(hasVersion({}, VERSION), false);
  assert.equal(hasVersion({ versions: null }, VERSION), false);
});

check("backoff grows but stays bounded", () => {
  const options = { baseSeconds: 5, capSeconds: 30 };
  assert.equal(backoffSeconds(1, options), 5);
  assert.equal(backoffSeconds(2, options), 10);
  assert.equal(backoffSeconds(3, options), 20);
  assert.equal(backoffSeconds(4, options), 30);
  assert.equal(backoffSeconds(11, options), 30);
  assert.equal(backoffSeconds(500, options), 30);
});

check("DIRECTION 1: 404-then-success resolves and stops on first match", async () => {
  // Four misses reproduces the original failure shape: the old loop died on the
  // sixth attempt at a flat 10s sleep, i.e. after roughly one minute.
  const { impl, state } = flakyFetch(4);
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 12,
    fetchImpl: impl,
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });

  assert.equal(result.visible, true, "the poll must report the version as visible");
  assert.equal(result.attempts, 5, "success must be detected on the first non-404 read");
  assert.equal(state.calls, 5, "polling must stop immediately after the first match");
  assert.ok(result.waitedSeconds >= 35, `waited ${result.waitedSeconds}s, expected the growing backoff`);
});

check("DIRECTION 2: a version that never appears still fails closed", async () => {
  const { impl, state } = flakyFetch(Number.MAX_SAFE_INTEGER);
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 12,
    fetchImpl: impl,
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });

  assert.equal(result.visible, false, "a never-published version must not be reported as visible");
  assert.equal(result.attempts, 12, "the poll must consume its whole budget");
  assert.equal(state.calls, 12, "no early exit on failure");
});

check("a transient network error keeps polling instead of failing the release", async () => {
  let calls = 0;
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 12,
    fetchImpl: async () => {
      calls += 1;
      if (calls <= 2) throw new Error("ECONNRESET");
      return { ok: true, status: 200, json: async () => ({ versions: { [VERSION]: { dist: {} } } }) };
    },
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });
  assert.equal(result.visible, true);
  assert.equal(calls, 3, "two network errors then success");
});

check("a 5xx is polled through, never treated as success", async () => {
  let calls = 0;
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 3,
    fetchImpl: async () => {
      calls += 1;
      return { ok: false, status: 503, json: async () => ({}) };
    },
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });
  assert.equal(result.visible, false);
  assert.equal(result.lastStatus, 503, "the last registry status must be reported");
  assert.equal(calls, 3);
});

check("a 200 that lacks the version keeps polling", async () => {
  let calls = 0;
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 3,
    fetchImpl: async () => {
      calls += 1;
      return { ok: true, status: 200, json: async () => ({ versions: { "2.0.0": {} } }) };
    },
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });
  assert.equal(result.visible, false, "a packument without the version is not success");
  assert.equal(calls, 3);
});

check("every read defeats caching in the URL and the headers", async () => {
  const { impl, state } = flakyFetch(1);
  await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 4,
    fetchImpl: impl,
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });
  for (const url of state.urls) {
    assert.match(url, /[?&]cb=/, `cache-buster missing from ${url}`);
  }
  for (const headers of state.headers) {
    assert.equal(headers["cache-control"], "no-cache");
    assert.equal(headers.pragma, "no-cache");
    assert.equal(headers.accept, "application/json");
  }
});

check("the default budget outlasts the propagation window and the cache TTL", async () => {
  const { impl } = flakyFetch(Number.MAX_SAFE_INTEGER);
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    fetchImpl: impl,
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });

  // Observed propagation ~95s in this org's sibling repositories.
  assert.ok(
    result.waitedSeconds >= 95,
    `budget ${result.waitedSeconds}s is under the ~95s propagation window that caused the false failure`,
  );
  // The registry serves the packument with `Cache-Control: public, max-age=300`,
  // so a shorter budget could still read a stale document.
  assert.ok(
    result.waitedSeconds >= 300,
    `budget ${result.waitedSeconds}s is under the registry's 300s cache TTL`,
  );
  assert.equal(result.attempts, 12, "the documented attempt count must not be shrunk");
});

check("the real registry answers a direct read for the published version", async () => {
  // Network-guarded so CI stays deterministic: skipped, never failed, offline.
  let reachable = true;
  try {
    await fetch("https://registry.npmjs.org/-/ping", { signal: AbortSignal.timeout(10_000) });
  } catch {
    reachable = false;
  }
  if (!reachable) {
    console.log("    (skipped live registry read: no network)");
    return;
  }
  const result = await waitForVisibility({
    pkg: PACKAGE,
    version: VERSION,
    attempts: 2,
    sleepImpl: noSleep,
    sleepScale: 0,
    log: quiet,
  });
  assert.equal(result.visible, true, "the registry should report the published version as visible");
});

check("CLI process exits 0 when the version becomes visible", () => {
  const run = runInChild(`
    let calls = 0;
    const fake = async () => {
      calls += 1;
      if (calls <= 3) return { ok: false, status: 404, json: async () => ({}) };
      return { ok: true, status: 200, json: async () => ({ versions: { "5.2.5": { dist: { tarball: "https://example.invalid/x.tgz" } } } }) };
    };
    const result = await waitForVisibility({
      pkg: "scientific-computing-system-2.0",
      version: "5.2.5",
      fetchImpl: fake,
      sleepImpl: async () => {},
      sleepScale: 0,
      log: () => {},
    });
    process.exit(result.visible ? 0 : 1);
  `);
  assert.equal(run.status, 0, `expected exit 0, got ${run.status}: ${run.stderr}`);
});

check("CLI process exits non-zero when the version never appears", () => {
  const run = runInChild(`
    const never = async () => ({ ok: false, status: 404, json: async () => ({}) });
    const result = await waitForVisibility({
      pkg: "scientific-computing-system-2.0",
      version: "5.2.5",
      attempts: 3,
      fetchImpl: never,
      sleepImpl: async () => {},
      sleepScale: 0,
      log: () => {},
    });
    process.exit(result.visible ? 0 : 1);
  `);
  assert.equal(run.status, 1, `expected exit 1, got ${run.status}: ${run.stderr}`);
});

check("a malformed package name is rejected rather than fetched", async () => {
  await assert.rejects(
    () =>
      waitForVisibility({
        pkg: "not a package",
        version: VERSION,
        fetchImpl: async () => {
          throw new Error("fetch must not be called");
        },
        sleepScale: 0,
        log: quiet,
      }),
    /invalid npm package name/,
  );
});

check("a non-positive attempt count is rejected", async () => {
  await assert.rejects(
    () =>
      waitForVisibility({
        pkg: PACKAGE,
        version: VERSION,
        attempts: 0,
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ versions: {} }) }),
        sleepScale: 0,
        log: quiet,
      }),
    /attempts must be a positive integer/,
  );
});

process.exit((await runChecks()) > 0 ? 1 : 0);