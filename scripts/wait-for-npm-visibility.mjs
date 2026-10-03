/**
 * Bounded poll for npm registry visibility after a publish.
 *
 * Why this exists: `npm publish` returns as soon as the registry's WRITE path
 * accepts the upload, but the READ path is served by a CDN that propagates
 * asynchronously. Observed propagation delays in this repository's sibling
 * projects are ~95s, and the packument is served with
 * `Cache-Control: public, max-age=300`, so a client-side cache can keep
 * replaying a stale document for up to five minutes.
 *
 * The previous implementation retried six times with a flat 10s sleep: a
 * ~60s budget that is both shorter than the propagation window and shorter
 * than the 300s cache TTL. A publish that had genuinely succeeded was
 * therefore reported as a hard failure (run 37120536500: attempts 1/6
 * through 6/6, then "was not visible on the registry after publishing", even
 * though 5.2.5 was in fact published).
 *
 * Why a direct registry read instead of `npm view`:
 *
 *   1. `npm view` resolves through npm's on-disk HTTP cache
 *      (`make-fetch-happen` + `cacache`, verified locally by reading the
 *      cache entry for the packument). That cache honours the registry's
 *      `max-age=300`, so retries inside the TTL can keep returning the
 *      pre-publish document. This script sends `Cache-Control: no-cache` and
 *      appends a cache-busting query parameter, which forces the CDN to
 *      revalidate instead of replaying the stale body.
 *   2. `npm view` conflates "not published yet" with "network/auth failure"
 *      in a single non-zero exit code, so the loop cannot distinguish a
 *      propagation delay from a broken registry. Reading the HTTP status
 *      directly lets a 404 keep polling while a 5xx is surfaced as a warning.
 *   3. One `node` process replacing N `npm view` subprocesses removes the
 *      per-attempt npm CLI startup cost from the poll budget.
 *
 * Contract:
 *   - First match wins: as soon as the version is present the script exits 0.
 *   - Bounded: `attempts` and a growing backoff cap the total wait.
 *   - Fails closed: exhausting the budget emits `::warning::` and exits
 *     non-zero, so a version that never appears is still a failure.
 *
 * Usage:
 *   node scripts/wait-for-npm-visibility.mjs <package> <version> [attempts]
 *
 * Environment:
 *   NPM_REGISTRY        registry base URL (default https://registry.npmjs.org)
 *   NPM_VISIBILITY_SLEEP_SCALE  multiplier for every sleep, for tests
 */

import { setTimeout as delay } from "node:timers/promises";

const PACKAGE_SCOPE = /^(@[a-z0-9-~][a-z0-9-._~]*\/)?[a-z0-9-~][a-z0-9-._~]*$/;

/** Exponential backoff with a hard cap, so the budget is bounded but generous. */
export function backoffSeconds(attempt, { baseSeconds = 5, capSeconds = 30 } = {}) {
  return Math.min(baseSeconds * 2 ** (attempt - 1), capSeconds);
}

/** True when `version` is a published version of `document`. */
export function hasVersion(document, version) {
  if (!document || typeof document !== "object") return false;
  const versions = document.versions;
  if (!versions || typeof versions !== "object") return false;
  return Object.prototype.hasOwnProperty.call(versions, version);
}

/**
 * Fetch the packument with caching defeated at both layers.
 *
 * The cache-buster is deliberately part of the URL: `Cache-Control: no-cache`
 * alone still lets a shared proxy answer from cache on some paths, whereas a
 * unique query string guarantees a distinct cache key at the CDN.
 *
 * The packument (`/<package>`) is read rather than the per-version document
 * (`/<package>/<version>`): only the packument carries the `versions` map, and
 * it is also the document `npm view` reads, so both agree on what "published"
 * means. The per-version document has no `versions` field at all.
 */
export async function readPackument({ registry, pkg, version, fetchImpl = fetch }) {
  const separator = registry.includes("?") ? "&" : "?";
  const url = `${registry}/${encodeURIComponent(pkg)}${separator}cb=${encodeURIComponent(version)}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const response = await fetchImpl(url, {
    headers: {
      accept: "application/json",
      "cache-control": "no-cache",
      pragma: "no-cache",
    },
  });

  if (response.status === 404) {
    return { visible: false, status: 404, reason: "not-found" };
  }
  if (!response.ok) {
    return { visible: false, status: response.status, reason: "registry-error" };
  }

  const document = await response.json();
  const visible = hasVersion(document, version);
  return { visible, status: response.status, reason: visible ? "visible" : "missing-version", document };
}

/**
 * Poll until the version is visible or the budget is exhausted.
 *
 * Returns { visible, attempts, waitedSeconds, lastStatus }. Never throws for an
 * ordinary "not there yet"; a thrown error is reserved for programmer error
 * such as a malformed package name.
 */
export async function waitForVisibility({
  pkg,
  version,
  attempts = 12,
  registry = process.env.NPM_REGISTRY ?? "https://registry.npmjs.org",
  fetchImpl = fetch,
  sleepImpl = delay,
  sleepScale = Number(process.env.NPM_VISIBILITY_SLEEP_SCALE ?? 1),
  baseSeconds = 5,
  capSeconds = 45,
  log = console.log,
}) {
  if (!PACKAGE_SCOPE.test(pkg)) {
    throw new Error(`invalid npm package name: ${pkg}`);
  }
  if (!version) {
    throw new Error("a version is required");
  }
  if (!Number.isInteger(attempts) || attempts < 1) {
    throw new Error(`attempts must be a positive integer, got ${attempts}`);
  }

  let waitedSeconds = 0;
  let lastStatus = 0;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    let result;
    try {
      result = await readPackument({ registry, pkg, version, fetchImpl });
    } catch (error) {
      // A transient network error is indistinguishable from propagation lag
      // from here, so it keeps polling rather than failing the release.
      log(`registry read failed (attempt ${attempt}/${attempts}): ${error.message}`);
      result = { visible: false, status: 0, reason: "network-error" };
    }
    lastStatus = result.status;

    if (result.visible) {
      log(`registry reports ${pkg}@${version} as published (attempt ${attempt}/${attempts})`);
      return { visible: true, attempts: attempt, waitedSeconds, lastStatus, document: result.document };
    }

    if (attempt === attempts) break;

    const seconds = backoffSeconds(attempt, { baseSeconds, capSeconds });
    waitedSeconds += seconds;
    log(
      `${pkg}@${version} not visible on the registry yet ` +
        `(attempt ${attempt}/${attempts}, status ${result.status ?? "network"}); ` +
        `retrying in ${seconds}s`,
    );
    if (sleepScale > 0) {
      await sleepImpl(Math.round(seconds * sleepScale * 1000));
    }
  }

  return { visible: false, attempts, waitedSeconds, lastStatus };
}

function parseArgv(argv) {
  const [pkg, version, attempts] = argv;
  if (!pkg || !version) {
    console.error("usage: node scripts/wait-for-npm-visibility.mjs <package> <version> [attempts]");
    process.exit(2);
  }
  return { pkg, version, attempts: attempts === undefined ? 12 : Number(attempts) };
}

const invokedDirectly = process.argv[1] && import.meta.url === `file://${process.argv[1].replace(/\\/g, "/")}`;
if (invokedDirectly) {
  const { pkg, version, attempts } = parseArgv(process.argv.slice(2));
  const result = await waitForVisibility({ pkg, version, attempts });

  if (result.visible) {
    const dist = result.document?.versions?.[version]?.dist ?? {};
    console.log(`verified npm publication: ${pkg}@${version}`);
    console.log(`tarball: ${dist.tarball ?? "unknown"}`);
    console.log(`integrity: ${dist.integrity ?? "unknown"}`);
    // Attestations are reported, never asserted. A publish made with a
    // long-lived automation token carries no Sigstore attestation at all, so
    // this workflow must not claim provenance it cannot prove.
    if (dist.attestations) {
      console.log(`attestations: ${JSON.stringify(dist.attestations)}`);
    } else {
      console.log("attestations: none published with this version (no Sigstore attestation)");
    }
    process.exit(0);
  }

  // Budget exhausted. Warn first so the log explains a slow-but-successful
  // publish, then fail closed so a never-published version still breaks CI.
  console.warn(
    `::warning::${pkg}@${version} was still not visible on the registry after ` +
      `${result.attempts} attempts (~${Math.round(result.waitedSeconds / 60)} min, ` +
      `last status ${result.lastStatus || "network"}). npm may still propagate it; ` +
      `re-check with \`npm view ${pkg}@${version} version\` before republishing, ` +
      `since npm versions are immutable.`,
  );
  console.error(`ERROR: ${pkg}@${version} was not visible on the registry within the poll budget`);
  process.exit(1);
}