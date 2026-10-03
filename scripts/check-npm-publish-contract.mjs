/**
 * Contract test for .github/workflows/npm-publish.yml.
 *
 * The publish workflow has two mutually exclusive credential paths and the
 * difference between them is security-relevant, so it is asserted here rather
 * than left to review:
 *
 *   oidc  (default, preferred) - OIDC trusted publishing, `npm publish
 *          --access public --provenance`, so the registry mints a Sigstore
 *          attestation for the publish.
 *   token (explicit opt-in)    - long-lived NPM_TOKEN. A classic automation
 *          token cannot mint a Sigstore attestation, so this path must NOT pass
 *          --provenance and must not claim provenance anywhere.
 *
 * Run via `npm test`.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..");
const workflowPath = join(repoRoot, ".github", "workflows", "npm-publish.yml");

const checks = [];
function check(name, fn) {
  try {
    fn();
    checks.push({ name, ok: true });
  } catch (error) {
    checks.push({ name, ok: false, error });
  }
}

const workflow = readFileSync(workflowPath, "utf8");

/** Return the YAML block of a single step that starts with `name: <name>`. */
function stepBlock(stepName) {
  const lines = workflow.split("\n");
  const startIndex = lines.findIndex(
    (line) => line.trim() === `- name: ${stepName}`,
  );
  assert.notEqual(startIndex, -1, `step not found: ${stepName}`);
  const indent = lines[startIndex].search(/\S/);
  let endIndex = lines.length;
  for (let i = startIndex + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.trim() === "") continue;
    const lineIndent = line.search(/\S/);
    if (lineIndent <= indent && line.trimStart().startsWith("- ")) {
      endIndex = i;
      break;
    }
  }
  return lines.slice(startIndex, endIndex).join("\n");
}

check("workflow file is readable", () => {
  assert.ok(workflow.length > 0, "workflow is empty");
});

check("use_token_fallback input exists and defaults to false", () => {
  assert.match(workflow, /use_token_fallback:/);
  assert.match(workflow, /type:\s*boolean/);
  const inputBlock = workflow.slice(
    workflow.indexOf("use_token_fallback:"),
    workflow.indexOf("concurrency:"),
  );
  assert.match(inputBlock, /required:\s*false/);
  assert.match(inputBlock, /default:\s*false/);
});

check("both publish modes are present and are distinct steps", () => {
  assert.match(workflow, /- name: Publish to npm \(OIDC trusted publishing\)/);
  assert.match(workflow, /- name: Publish to npm \(long-lived NPM_TOKEN\)/);
  assert.notEqual(
    stepBlock("Publish to npm (OIDC trusted publishing)"),
    stepBlock("Publish to npm (long-lived NPM_TOKEN)"),
  );
});

check("oidc path publishes with --provenance", () => {
  const block = stepBlock("Publish to npm (OIDC trusted publishing)");
  assert.match(block, /npm publish --access public --provenance/);
});

check("token path publishes WITHOUT --provenance", () => {
  const block = stepBlock("Publish to npm (long-lived NPM_TOKEN)");
  assert.match(block, /npm publish --access public/);
  assert.ok(
    !block.includes("--provenance"),
    "token path must not pass --provenance: a long-lived token cannot mint a Sigstore attestation",
  );
});

check("token path supplies NODE_AUTH_TOKEN from secrets.NPM_TOKEN", () => {
  const block = stepBlock("Publish to npm (long-lived NPM_TOKEN)");
  assert.match(block, /NODE_AUTH_TOKEN:\s*\$\{\{\s*secrets\.NPM_TOKEN\s*\}\}/);
});

check("each publish step is gated on its own mode", () => {
  assert.match(
    stepBlock("Publish to npm (OIDC trusted publishing)"),
    /steps\.mode\.outputs\.mode\s*==\s*'oidc'/,
  );
  assert.match(
    stepBlock("Publish to npm (long-lived NPM_TOKEN)"),
    /steps\.mode\.outputs\.mode\s*==\s*'token'/,
  );
});

check("requested mode without a credential fails closed", () => {
  const block = stepBlock("Resolve publish mode");
  assert.match(block, /exit 1/, "mode resolution must be able to fail closed");
  assert.match(block, /if \[ -z "\$\{NPM_TOKEN\}" \]/);
  assert.match(block, /use_token_fallback=true was requested but the NPM_TOKEN secret is empty/);
});

check("oidc remains the default when no input is supplied", () => {
  const block = stepBlock("Resolve publish mode");
  // An empty input falls through to OIDC; only an explicit "true" selects token.
  assert.match(block, /false\|""\)\s*\n\s*MODE=oidc/);
  assert.ok(
    !/^\s*true\)\s*\n\s*MODE=oidc/m.test(block),
    "an empty input must not select the token path",
  );
});

check("an explicit use_token_fallback=false is accepted, not rejected", () => {
  // Regression guard: the case arm must accept "false" as a valid boolean.
  // An earlier revision only matched `true` and `""`, so dispatching with an
  // explicit false failed with "must be a boolean".
  const block = stepBlock("Resolve publish mode");
  assert.match(
    block,
    /false\|""\)/,
    "the false|empty case arm is missing; an explicit false would be rejected",
  );
});

check("npm version floor is asserted numerically, not by regex", () => {
  // A pattern such as ^11\.(5[1-9]|[6-9][0-9])\. wrongly rejects npm 11.19.0,
  // which is what Node 24 bundles, because [6-9][0-9] only covers 11.60-11.99.
  // Scan executable lines only, so the comment documenting that pitfall does
  // not trip the check while a real regex gate still would.
  const executable = workflow
    .split("\n")
    .filter((line) => !/^\s*#/.test(line))
    .join("\n");
  assert.ok(
    !/5\[1-9\]/.test(executable),
    "regex-based npm version gate reintroduced",
  );
  assert.ok(
    !/\[6-9\]\[0-9\]/.test(executable),
    "regex-based npm version gate reintroduced",
  );
  const block = stepBlock("Fail closed when npm is too old for OIDC trusted publishing");
  assert.match(block, /major > 11 \|\| \(major === 11 && minor >= 5\)/);
  assert.match(block, /split\('\.'\)\.map\(Number\)/);
  // The pitfall stays documented for the next person who "simplifies" it.
  assert.match(
    workflow,
    /11\.19\.0/,
    "the 11.19.0 pitfall should stay documented in the workflow",
  );
});

check("the OIDC npm floor only applies to the OIDC path", () => {
  const block = stepBlock("Fail closed when npm is too old for OIDC trusted publishing");
  assert.match(block, /if: steps\.mode\.outputs\.mode == 'oidc'/);
});

check("provenance is not claimed on the token path", () => {
  const block = stepBlock("Verify npm publication");
  assert.match(
    block,
    /no Sigstore provenance attestation exists on this path/,
    "verification must state that the token path has no attestation",
  );
  assert.match(block, /PUBLISH_MODE/);
  // attestations may only be reported in the OIDC branch
  const attestationsIndex = block.indexOf("dist.attestations");
  const oidcBranchIndex = block.indexOf('= "oidc"');
  assert.ok(attestationsIndex > oidcBranchIndex, "attestations must be OIDC-only");
});

check("existing gates are retained", () => {
  assert.match(workflow, /node --check index\.js/);
  assert.match(workflow, /node --check bin\/scs2\.js/);
  assert.match(workflow, /npm test/);
  assert.match(workflow, /tarball contents match the files allowlist/);
  assert.match(workflow, /version lockstep ok/);
  assert.match(workflow, /Check whether this version already exists on npm/);
});

let failed = 0;
for (const { name, ok, error } of checks) {
  if (ok) {
    console.log(`ok - ${name}`);
  } else {
    failed += 1;
    console.error(`not ok - ${name}`);
    console.error(`    ${error.message.split("\n").join("\n    ")}`);
  }
}
console.log(`\n${checks.length - failed}/${checks.length} contract checks passed`);
if (failed > 0) process.exit(1);