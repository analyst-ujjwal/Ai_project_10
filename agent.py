"""
groq_proactive_agent.py

A modular AI agent using Groq's Python SDK + LLaMA models.
- Includes 18 example tools (local implementations or placeholders).
- Proactive behavior: initialized with a system prompt that encourages initiative.

USAGE:
1) pip install -r requirements.txt
2) export GROQ_API_KEY="your_api_key_here"
3) python groq_proactive_agent.py

Author: generated for UJJWAL
"""

import os
import json
import time
import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ====== Load environment ======
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in .env or environment variables")

MODEL_NAME = "llama-3.1-8b-instant"

# ====== Groq client ======
client = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.7,
    max_tokens=512,
)

# ====== Tool base classes ======
class ToolResponse:
    def __init__(self, success: bool, output: Any, metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.output = output
        self.metadata = metadata or {}

    def to_dict(self):
        return {"success": self.success, "output": self.output, "metadata": self.metadata}


class Tool:
    name: str = "base"
    def run(self, **kwargs) -> ToolResponse:
        raise NotImplementedError


# ====== Helper for safe Groq calls ======
def groq_call(prompt: str) -> str:
    """Wrapper to safely invoke Groq model and extract text."""
    try:
        resp = client.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"[Groq error: {e}]"


# ====== Tool implementations ======

class WebSearchTool(Tool):
    name = "web_search"
    def run(self, query: str, top_k: int = 3) -> ToolResponse:
        prompt = (
            f"You are a web researcher. Provide {top_k} likely current results for '{query}', "
            "formatted as JSON: {\"results\": [{\"title\":..., \"snippet\":..., \"url\":...}]}"
        )
        return ToolResponse(True, groq_call(prompt))


class CalculatorTool(Tool):
    name = "calculator"
    def run(self, expression: str) -> ToolResponse:
        try:
            result = eval(expression, {"__builtins__": None}, {})
            return ToolResponse(True, result)
        except Exception as e:
            return ToolResponse(False, str(e))


class FileSystemTool(Tool):
    name = "filesystem"
    def run(self, action: str, path: str, content: Optional[str] = None) -> ToolResponse:
        try:
            if action == "read":
                with open(path, "r", encoding="utf-8") as f:
                    return ToolResponse(True, f.read())
            elif action == "write":
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content or "")
                return ToolResponse(True, f"Wrote to {path}")
            else:
                return ToolResponse(False, "Unknown action")
        except Exception as e:
            return ToolResponse(False, str(e))


class EmailTool(Tool):
    name = "email"
    def run(self, to: str, subject: str, body: str) -> ToolResponse:
        print(f"[EMAIL SIMULATION]\nTo: {to}\nSubject: {subject}\nBody: {body}\n")
        return ToolResponse(True, "Email simulated")


class TerminalTool(Tool):
    name = "terminal"
    def run(self, command: str, dry_run: bool = True) -> ToolResponse:
        if dry_run:
            return ToolResponse(True, f"Dry-run: would execute '{command}'")
        import subprocess
        try:
            out = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
            return ToolResponse(True, out)
        except Exception as e:
            return ToolResponse(False, str(e))


class CalendarTool(Tool):
    name = "calendar"
    def __init__(self):
        self.events = []
    def run(self, action: str, title: str = "", time_iso: str = "") -> ToolResponse:
        if action == "add":
            self.events.append({"title": title, "time": time_iso})
            return ToolResponse(True, f"Added event '{title}'")
        elif action == "list":
            return ToolResponse(True, self.events)
        return ToolResponse(False, "Unknown action")


class NoteTool(Tool):
    name = "notes"
    def __init__(self):
        self.notes = []
    def run(self, note: str) -> ToolResponse:
        ts = datetime.datetime.utcnow().isoformat()
        self.notes.append({"ts": ts, "note": note})
        return ToolResponse(True, f"Saved note at {ts}")


class SummarizerTool(Tool):
    name = "summarizer"
    def run(self, text: str, max_sentences: int = 4) -> ToolResponse:
        prompt = f"Summarize in {max_sentences} sentences:\n{text}"
        return ToolResponse(True, groq_call(prompt))


class FetchTool(Tool):
    name = "fetch"
    def run(self, url: str) -> ToolResponse:
        prompt = f"Pretend you fetched {url}. Provide plausible title, summary, and 3 things to verify."
        return ToolResponse(True, groq_call(prompt))


class PythonRunnerTool(Tool):
    name = "python_runner"
    def run(self, code: str) -> ToolResponse:
        try:
            safe_globals = {"__builtins__": None}
            local_vars = {}
            exec(code, safe_globals, local_vars)
            return ToolResponse(True, local_vars)
        except Exception as e:
            return ToolResponse(False, str(e))


class ImageGenTool(Tool):
    name = "image_gen"
    def run(self, prompt: str, size: str = "512x512") -> ToolResponse:
        return ToolResponse(True, f"Generated image placeholder for: '{prompt}' ({size})")


class DBTool(Tool):
    name = "db"
    def __init__(self, filename: str = "agent_db.json"):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump({}, f)
    def run(self, action: str, key: str = "", value: Any = None) -> ToolResponse:
        with open(self.filename, "r") as f:
            db = json.load(f)
        if action == "get":
            return ToolResponse(True, db.get(key))
        elif action == "set":
            db[key] = value
            with open(self.filename, "w") as f:
                json.dump(db, f, indent=2)
            return ToolResponse(True, "OK")
        elif action == "list":
            return ToolResponse(True, db)
        return ToolResponse(False, "Unknown DB action")


class SchedulerTool(Tool):
    name = "scheduler"
    def run(self, task_name: str, run_after_seconds: int = 1) -> ToolResponse:
        time.sleep(min(run_after_seconds, 1))
        return ToolResponse(True, f"Executed '{task_name}' after {run_after_seconds}s (simulated)")


class TranslateTool(Tool):
    name = "translate"
    def run(self, text: str, target_lang: str = "en") -> ToolResponse:
        prompt = f"Translate this to {target_lang}:\n{text}"
        return ToolResponse(True, groq_call(prompt))


class SentimentTool(Tool):
    name = "sentiment"
    def run(self, text: str) -> ToolResponse:
        prompt = f"Determine sentiment and reason for: {text}"
        return ToolResponse(True, groq_call(prompt))


class ResearchPlanTool(Tool):
    name = "research_plan"
    def run(self, topic: str, depth: int = 3) -> ToolResponse:
        prompt = f"Create a {depth}-step research plan for: {topic}"
        return ToolResponse(True, groq_call(prompt))


class ProactivityTool(Tool):
    name = "proactivity"
    def run(self, context: str) -> ToolResponse:
        prompt = (
            "You are a meta-reasoner. Based on the context, decide JSON {\"decision\": \"act\"|\"ask\"|\"wait\", \"reason\": \"...\"}.\n\n"
            f"Context:\n{context}"
        )
        return ToolResponse(True, groq_call(prompt))


class LoggerTool(Tool):
    name = "logger"
    def run(self, level: str, message: str) -> ToolResponse:
        ts = datetime.datetime.utcnow().isoformat()
        with open("agent.log", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {level.upper()}: {message}\n")
        return ToolResponse(True, "Logged")


# ====== Agent ======
class Agent:
    def __init__(self):
        self.tools = {
            t.name: t for t in [
                WebSearchTool(), CalculatorTool(), FileSystemTool(), EmailTool(), TerminalTool(),
                CalendarTool(), NoteTool(), SummarizerTool(), FetchTool(), PythonRunnerTool(),
                ImageGenTool(), DBTool(), SchedulerTool(), TranslateTool(), SentimentTool(),
                ResearchPlanTool(), ProactivityTool(), LoggerTool()
            ]
        }
        self.system_prompt = (
            "You are a proactive AI agent. Take initiative when safe, ask if unsure, and log actions."
        )

    def call_model(self, user_message: str) -> str:
        prompt = f"{self.system_prompt}\nUser: {user_message}"
        return groq_call(prompt)

    def use_tool(self, tool_name: str, **kwargs) -> ToolResponse:
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResponse(False, f"Unknown tool: {tool_name}")
        return tool.run(**kwargs)

    def handle(self, instruction: str) -> Dict[str, Any]:
        p = self.use_tool("proactivity", context=instruction)
        self.use_tool("logger", level="info", message=f"Decision: {p.output}")
        if "act" in p.output:
            plan = self.use_tool("research_plan", topic=instruction)
            search = self.use_tool("web_search", query=instruction)
            summary = self.use_tool("summarizer", text=str(search.output))
            note = self.use_tool("notes", note=f"Auto-summary: {summary.output}")
            return {
                "decision": "act",
                "plan": plan.output,
                "search": search.output,
                "summary": summary.output,
                "note": note.output,
            }
        else:
            reply = self.call_model(f"Instruction: {instruction}. Suggest next actions.")
            return {"decision": "ask", "proposal": reply}


# ====== CLI ======
if __name__ == "__main__":
    agent = Agent()
    print("Groq Proactive Agent — interactive mode")
    print("Try: 'Research best vector databases' or 'Translate hello to French'\n")
    while True:
        user_in = input("You: ").strip()
        if user_in.lower() in ("exit", "quit"):
            break
        out = agent.handle(user_in)
        print("Agent:", json.dumps(out, indent=2))
