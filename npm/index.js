#!/usr/bin/env node
/**
 * kredo-mcp: run kredo decision models as MCP tools, zero install.
 *
 *   npx kredo-mcp                       # run the MCP server (downloads the binary on first use)
 *   npx kredo-mcp install               # register with Claude Code, Claude Desktop and opencode
 *   npx kredo-mcp install claude        # only Claude Code + Desktop
 *   npx kredo-mcp install opencode      # only opencode
 *
 * The kredo binary is fetched from GitHub Releases into ~/.kredo/bin and
 * reused across runs. Set KREDO_BIN to skip the download and use a local
 * build.
 */

"use strict";

const { execFileSync, spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const https = require("https");

const VERSION = require("./package.json").version;
const REPO = "einyx/kredo";
const BIN_DIR = path.join(os.homedir(), ".kredo", "bin");

function targetTriple() {
  const platform = os.platform();
  const arch = os.arch();
  if (platform === "darwin" && arch === "arm64") return "aarch64-apple-darwin";
  if (platform === "darwin") return "x86_64-apple-darwin";
  if (platform === "linux" && arch === "arm64") return "aarch64-unknown-linux-gnu";
  if (platform === "linux") return "x86_64-unknown-linux-gnu";
  console.error(`kredo-mcp: unsupported platform ${platform}-${arch}`);
  process.exit(1);
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const get = (u, redirects) => {
      if (redirects > 5) return reject(new Error("too many redirects"));
      https
        .get(u, (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            return get(res.headers.location, redirects + 1);
          }
          if (res.statusCode !== 200) {
            return reject(new Error(`HTTP ${res.statusCode} for ${u}`));
          }
          const file = fs.createWriteStream(dest);
          res.pipe(file);
          file.on("finish", () => file.close(resolve));
          file.on("error", reject);
        })
        .on("error", reject);
    };
    get(url, 0);
  });
}

async function resolveBinary() {
  if (process.env.KREDO_BIN && fs.existsSync(process.env.KREDO_BIN)) {
    return process.env.KREDO_BIN;
  }
  // Prefer an existing local install.
  try {
    const which = spawnSync(
      process.platform === "win32" ? "where" : "which",
      ["kredo"],
      { encoding: "utf8" },
    );
    if (which.status === 0 && which.stdout.trim()) return which.stdout.trim().split("\n")[0];
  } catch {}

  fs.mkdirSync(BIN_DIR, { recursive: true });
  const target = targetTriple();
  const dest = path.join(BIN_DIR, `kredo-v${VERSION}-${target}`);
  if (fs.existsSync(dest)) return dest;

  const url = `https://github.com/${REPO}/releases/download/v${VERSION}/kredo-v${VERSION}-${target}.tar.gz`;
  const tgz = dest + ".tar.gz";
  console.error(`kredo-mcp: downloading kredo v${VERSION} (${target})…`);
  await download(url, tgz);
  // Extract into a scratch dir; the tarball contains kredo-v<ver>-<target>/kredo
  const scratch = path.join(BIN_DIR, `.extract-${Date.now()}`);
  fs.mkdirSync(scratch, { recursive: true });
  execFileSync("tar", ["xzf", tgz, "-C", scratch]);
  fs.renameSync(path.join(scratch, `kredo-v${VERSION}-${target}`, "kredo"), dest);
  fs.rmSync(scratch, { recursive: true, force: true });
  fs.rmSync(tgz, { force: true });
  fs.chmodSync(dest, 0o755);
  return dest;
}

/* ---------- MCP config registration ---------- */

function readJson(p) {
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return {};
  }
}
function writeJson(p, v) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(v, null, 2) + "\n");
}

function serverEntry(bin) {
  return { command: bin, args: ["mcp"] };
}

function registerClaudeCode(bin) {
  const cfgPath = path.join(os.homedir(), ".claude.json");
  const cfg = readJson(cfgPath);
  cfg.mcpServers = cfg.mcpServers || {};
  cfg.mcpServers.kredo = { type: "stdio", ...serverEntry(bin) };
  writeJson(cfgPath, cfg);
  return "Claude Code (~/.claude.json)";
}

function registerClaudeDesktop(bin) {
  const cfgPath =
    process.platform === "darwin"
      ? path.join(os.homedir(), "Library", "Application Support", "Claude", "claude_desktop_config.json")
      : path.join(os.homedir(), ".config", "Claude", "claude_desktop_config.json");
  const cfg = readJson(cfgPath);
  cfg.mcpServers = cfg.mcpServers || {};
  cfg.mcpServers.kredo = serverEntry(bin);
  writeJson(cfgPath, cfg);
  return "Claude Desktop";
}

function registerOpencode(bin) {
  const cfgPath = path.join(os.homedir(), ".config", "opencode", "opencode.json");
  const cfg = readJson(cfgPath);
  cfg.mcp = cfg.mcp || {};
  cfg.mcp.kredo = { type: "local", command: [bin, "mcp"], enabled: true };
  writeJson(cfgPath, cfg);
  return "opencode (~/.config/opencode/opencode.json)";
}

async function install(clients) {
  const bin = await resolveBinary();
  const all = { claude: registerClaudeCode, desktop: registerClaudeDesktop, opencode: registerOpencode };
  const wanted = clients.length ? clients : Object.keys(all);
  for (const c of wanted) {
    const fn = all[c];
    if (!fn) {
      console.error(`unknown client '${c}' (claude | desktop | opencode)`);
      continue;
    }
    console.log(`registered ${fn(bin)}`);
  }
  console.log("\nstart the daemon with:  kredo serve   (the MCP tools talk to it)");
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === "install") {
    await install(rest.flatMap((c) => c.split(",")));
    return;
  }
  if (cmd === "--help" || cmd === "-h" || cmd === "help") {
    console.log(
      "usage: npx kredo-mcp [install [claude,desktop,opencode]]\n" +
        "  (no command)   run the kredo MCP server over stdio\n" +
        "  install        register the kredo MCP server with MCP clients",
    );
    return;
  }
  // Default: act as the MCP server (for "npx -y kredo-mcp" configs).
  const bin = await resolveBinary();
  const result = spawnSync(bin, ["mcp", ...process.argv.slice(2)], { stdio: "inherit" });
  process.exit(result.status ?? 0);
}

main().catch((e) => {
  console.error("kredo-mcp:", e.message);
  process.exit(1);
});
