import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough, Writable } from "node:stream";
import test from "node:test";
import { runSetupWizard } from "./lib/setup-wizard.mjs";

const offlineFetch = async () => {
  throw new Error("offline test");
};

/**
 * Drive the wizard with scripted answers: readline writes a prompt without a
 * trailing newline, so every such chunk gets the next answer fed back.
 */
function scriptedIo(answers) {
  const queue = [...answers];
  const input = new PassThrough();
  const written = [];
  const output = new Writable({
    write(chunk, _enc, cb) {
      const text = chunk.toString();
      written.push(text);
      if (!text.endsWith("\n")) {
        setImmediate(() => input.write(`${queue.shift() ?? ""}\n`));
      }
      cb();
    },
  });
  return { input, output, transcript: () => written.join("") };
}

test("a missing config file is created at the candidate path", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-new-"));
  const target = join(dir, "nested", "ovcli.conf");
  try {
    const io = scriptedIo(["1", "sk-test", "y"]);
    const result = await runSetupWizard({
      input: io.input,
      output: io.output,
      env: { OPENVIKING_CLI_CONFIG_FILE: target },
      fetchImpl: offlineFetch,
    });

    assert.equal(result.written, true);
    assert.equal(result.path, target);
    assert.match(io.transcript(), new RegExp(`Config file: ${target}`));
    assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
      url: "http://127.0.0.1:1933",
      api_key: "sk-test",
      account: "default",
      user: "default",
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("a missing API key is rejected until a non-empty key is entered", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-key-required-"));
  const target = join(dir, "ovcli.conf");
  try {
    const io = scriptedIo(["1", "", "sk-required", "y"]);
    const result = await runSetupWizard({
      input: io.input,
      output: io.output,
      env: { OPENVIKING_CLI_CONFIG_FILE: target },
      fetchImpl: offlineFetch,
    });

    assert.equal(result.written, true);
    assert.match(io.transcript(), /API key is required/);
    assert.equal(JSON.parse(await readFile(target, "utf-8")).api_key, "sk-required");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("an existing config file keeps unknown fields and its secret", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-existing-"));
  const target = join(dir, "ovcli.conf");
  try {
    await writeFile(
      target,
      JSON.stringify({ url: "https://old.example.com", api_key: "sk-old", account: "acct" }),
    );
    const io = scriptedIo(["1", "", "y"]);
    const result = await runSetupWizard({
      input: io.input,
      output: io.output,
      env: { OPENVIKING_CLI_CONFIG_FILE: target },
      fetchImpl: offlineFetch,
    });

    assert.equal(result.path, target);
    assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
      url: "http://127.0.0.1:1933",
      api_key: "sk-old",
      account: "default",
      user: "default",
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("a trusted server is probed with the collected account and user", async (t) => {
  const requests = [];
  const server = createServer((req, res) => {
    requests.push({ path: req.url, headers: req.headers });
    if (req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ healthy: true, auth_mode: "trusted" }));
      return;
    }
    const valid = req.headers.authorization === "Bearer root-key"
      && req.headers["x-openviking-account"] === "acme"
      && req.headers["x-openviking-user"] === "alice";
    res.writeHead(valid ? 200 : 401, { "Content-Type": "application/json" });
    res.end(JSON.stringify(valid
      ? { status: "ok", result: { initialized: true } }
      : { error: { message: "invalid trusted identity" } }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-trusted-"));
  const target = join(dir, "ovcli.conf");
  t.after(() => rm(dir, { recursive: true, force: true }));
  const url = `http://127.0.0.1:${server.address().port}`;
  const io = scriptedIo(["3", url, "root-key", "acme", "alice", "y"]);

  const result = await runSetupWizard({
    input: io.input,
    output: io.output,
    env: { OPENVIKING_CLI_CONFIG_FILE: target },
  });

  assert.equal(result.written, true);
  assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
    url,
    api_key: "root-key",
    account: "acme",
    user: "alice",
  });
  assert.equal(requests[0].path, "/health");
  assert.equal(requests[0].headers.authorization, undefined);
  assert.ok(requests.some((request) =>
    request.path === "/api/v1/system/status"
    && request.headers.authorization === "Bearer root-key"
    && request.headers["x-openviking-account"] === "acme"
    && request.headers["x-openviking-user"] === "alice"
  ));
});

test("a loopback API key server does not add trusted identity", async (t) => {
  const server = createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ healthy: true, auth_mode: "api_key" }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-api-key-"));
  const target = join(dir, "ovcli.conf");
  t.after(() => rm(dir, { recursive: true, force: true }));
  const url = `http://127.0.0.1:${server.address().port}`;
  const io = scriptedIo(["3", url, "user-key", "y"]);

  const result = await runSetupWizard({
    input: io.input,
    output: io.output,
    env: { OPENVIKING_CLI_CONFIG_FILE: target },
  });

  assert.equal(result.written, true);
  assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
    url,
    api_key: "user-key",
  });
});

test("an offline custom remote can be explicitly configured as trusted", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-offline-trusted-"));
  const target = join(dir, "ovcli.conf");
  t.after(() => rm(dir, { recursive: true, force: true }));
  const io = scriptedIo([
    "3",
    "https://offline.example.com",
    "root-key",
    "2",
    "acme",
    "alice",
    "y",
  ]);

  const result = await runSetupWizard({
    input: io.input,
    output: io.output,
    env: { OPENVIKING_CLI_CONFIG_FILE: target },
    fetchImpl: offlineFetch,
  });

  assert.equal(result.written, true);
  assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
    url: "https://offline.example.com",
    api_key: "root-key",
    account: "acme",
    user: "alice",
  });
});

test("switching from a remote profile to local resets identity to local defaults", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-local-reset-"));
  const target = join(dir, "ovcli.conf");
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(target, JSON.stringify({
    url: "https://old.example.com",
    api_key: "old-key",
    account: "acme",
    user: "alice",
  }));
  const io = scriptedIo(["1", "", "y"]);

  await runSetupWizard({
    input: io.input,
    output: io.output,
    env: { OPENVIKING_CLI_CONFIG_FILE: target },
    fetchImpl: offlineFetch,
  });

  assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
    url: "http://127.0.0.1:1933",
    api_key: "old-key",
    account: "default",
    user: "default",
  });
});

test("switching from trusted to API key removes the old identity", async (t) => {
  const server = createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ healthy: true, auth_mode: "api_key" }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-api-key-reset-"));
  const target = join(dir, "ovcli.conf");
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(target, JSON.stringify({
    url: "https://old.example.com",
    api_key: "old-key",
    account: "acme",
    user: "alice",
  }));
  const url = `http://127.0.0.1:${server.address().port}`;
  const io = scriptedIo(["3", url, "", "y"]);

  await runSetupWizard({
    input: io.input,
    output: io.output,
    env: { OPENVIKING_CLI_CONFIG_FILE: target },
  });

  assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
    url,
    api_key: "old-key",
  });
});

test("rerunning local setup preserves an explicit local trusted identity", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-local-preserve-"));
  const target = join(dir, "ovcli.conf");
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(target, JSON.stringify({
    url: "http://127.0.0.1:1933",
    api_key: "root-key",
    account: "acme",
    user: "alice",
  }));
  const io = scriptedIo(["1", "", "y"]);

  await runSetupWizard({
    input: io.input,
    output: io.output,
    env: { OPENVIKING_CLI_CONFIG_FILE: target },
    fetchImpl: offlineFetch,
  });

  assert.deepEqual(JSON.parse(await readFile(target, "utf-8")), {
    url: "http://127.0.0.1:1933",
    api_key: "root-key",
    account: "acme",
    user: "alice",
  });
});

test("declining the confirmation writes nothing but still names the target", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ov-wizard-abort-"));
  const target = join(dir, "missing", "ovcli.conf");
  try {
    await mkdir(join(dir, "missing"), { recursive: true });
    const io = scriptedIo(["2", "sk-test", "n"]);
    const result = await runSetupWizard({
      input: io.input,
      output: io.output,
      env: { OPENVIKING_CLI_CONFIG_FILE: target },
      fetchImpl: offlineFetch,
    });

    assert.equal(result.written, false);
    assert.equal(result.path, target);
    await assert.rejects(() => readFile(target, "utf-8"));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("the pi experimental generated copy matches the shared wizard", async () => {
  const source = await readFile(new URL("./lib/setup-wizard.mjs", import.meta.url), "utf-8");
  const generated = await readFile(
    new URL("../pi-experimental-context-management/shared/setup-wizard.mjs", import.meta.url),
    "utf-8",
  );

  assert.equal(
    generated.replace(
      "// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.\n",
      "",
    ),
    source,
  );
});
