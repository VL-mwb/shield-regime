# 🛡️ ShieldRegime Agency & MCP Server

Welcome to the **Agent-Native** directory of ShieldRegime. 

In the era of autonomous AI agents and automated algorithmic trading, financial compliance cannot rely solely on human audits. This module equips Large Language Models (LLMs) and autonomous AI agents across different ecosystems with the ability to natively perform physics-inspired market manipulation detection.

## 📂 Framework Integrations

ShieldRegime provides pre-built system prompts and rules tailored for the three most prominent AI Agent paradigms:

### 1. Antigravity / AutoGen (`antigravity/SKILL.md`)
Standard YAML-frontmatter skill definition designed for autonomous agent frameworks (like Antigravity or Microsoft AutoGen). Drop this file into your agent's `skills` folder to instantly grant it the ability to perform rigorous mathematical compliance audits.

### 2. Claude Code (`claude_code/market_surveillance.prompt`)
A specialized XML-structured prompt designed for Anthropic's Claude Code terminal agent. This teaches Claude how to act as a real-time compliance officer and leverage the MCP server directly from your terminal.

### 3. Cursor / Codex IDE (`cursor/.cursorrules`)
System instructions designed for AI-powered IDEs (Cursor, GitHub Copilot). Drop this `.cursorrules` file into your trading project's root directory, and your AI assistant will automatically enforce ShieldRegime compliance checks whenever it generates trading code!

---

## 🔌 Universal Backend: Model Context Protocol (MCP)

No matter which agent framework you use, they all need to execute code securely. 
We provide `mcp_server.py`, a lightweight **Model Context Protocol (MCP)** server.

[MCP](https://modelcontextprotocol.io/) is an open standard that enables AI models to securely interact with local Python tools. By running the `mcp_server.py`, your AI assistant can seamlessly call the `scan_ticker_for_manipulation` tool.

### Setup for Claude Desktop / Claude Code
Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "shield-regime": {
      "command": "python",
      "args": [
        "/absolute/path/to/shield-regime/agency/mcp_server.py"
      ]
    }
  }
}
```
