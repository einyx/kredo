# kredo-mcp

kredo decision models as [MCP](https://modelcontextprotocol.io) tools — for
Claude Code, Claude Desktop, opencode and any MCP client. Zero install:

```sh
# register with Claude Code, Claude Desktop and opencode
npx kredo-mcp install

# or just opencode / just claude
npx kredo-mcp install opencode
npx kredo-mcp install claude

# or run the MCP server directly
npx kredo-mcp
```

The first run downloads the kredo binary (~17 MB) from GitHub Releases into
`~/.kredo/bin` and reuses it. Tools: `kredo_decide`, `kredo_list_models`,
`kredo_describe_model`.

Start the daemon so the tools have something to talk to:

```sh
kredo serve
```

Set `KREDO_BIN=/path/to/kredo` to use a local build instead of the
downloaded release.
