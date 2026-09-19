# ================================================================
# GRAC V6
# Private Local AI Desktop
#
# Permanent entry point: AI.py
#
# Features
# ------------------------------------------------
# • Native PySide6 desktop application
# • Ollama streaming
# • Llama 3.2 3B
# • Persistent SQLite conversations
# • Persistent long-term memories
# • Faster limited-context generation
# • Animated processing states
# • Rich Markdown-ish responses
# • Beautiful formula / code cards
# • Copy buttons
# • Image attachments
# • PDF attachments
# • Text-file attachments
# • Attachment previews
# • PDF text extraction
# • Optional Ollama vision routing
# • Model selector
# • Search
# • Rename/delete conversations
# • Memory manager
# • Stop generation
# • Local/private status
#
# ================================================================

import sys
import re
import json
import html
import sqlite3
import base64
import random
from datetime import datetime
from pathlib import Path

import requests
import queue

try:
    import openvino_genai
    from PIL import Image
except ImportError:
    openvino_genai = None
    Image = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from PySide6.QtCore import Qt, Signal, QThread, QTimer, QUrl
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QTextBrowser,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QMenu,
    QMessageBox,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QFileDialog,
    QComboBox,
)


# ================================================================
# CONFIGURATION
# ================================================================

APP_NAME = "GRAC"
APP_VERSION = "6.3"

DATABASE = "grac.db"

OLLAMA_URL = "http://localhost:11434/api/chat"

# Main brain
DEFAULT_MODEL = "llama3.2:3b"

# You can later install a vision model and change this.
VISION_MODEL = "gemma3:4b"

MAX_HISTORY_MESSAGES = 12
MAX_PDF_CHARS = 18000
MAX_TEXT_ATTACHMENT_CHARS = 18000

KEEP_ALIVE = "30m"

OLLAMA_OPTIONS = {
    "num_ctx": 4096
}

SYSTEM_PROMPT = """
You are GRAC, a private personal AI assistant running locally.

IDENTITY
You are part of the GRAC Personal AI system.

BEHAVIOUR
- Be natural and conversational.
- Give direct answers.
- Prefer useful structure over huge walls of text.
- Use headings when they genuinely improve readability.
- Use bullet points for groups of facts.
- Keep simple questions concise.
- Explain technical subjects clearly.
- Use supplied long-term memories only when relevant.
- Never invent memories.
- Never claim to remember information not provided to you.
- Never expose private chain-of-thought.

FORMATTING
GRAC's interface supports lightweight Markdown.

Use:
# Heading
## Heading
**bold**
`inline code`
```language
code
```

For important standalone mathematical formulas, put the formula
on its own line surrounded by $$.

Example:

$$ P = VI $$

Do not surround ordinary sentences with $$.
"""


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / DATABASE
ATTACHMENT_DIR = BASE_DIR / "attachments"
GENERATED_DIR = BASE_DIR / "generated"

ATTACHMENT_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

IMAGE_MODEL_DIR = BASE_DIR / "models" / "image" / "sd15"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_STEPS = 15
IMAGE_DEVICE = "GPU"


# ================================================================
# DATABASE
# ================================================================

def db_connect():
    db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    db.execute("PRAGMA foreign_keys = ON")
    return db


def setup_database():
    db = db_connect()

    db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            message_id INTEGER,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.commit()
    db.close()


# ================================================================
# CHAT DATABASE
# ================================================================

def create_chat():
    db = db_connect()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO conversations(title, created_at) VALUES (?, ?)",
        ("New chat", datetime.now().isoformat()),
    )
    chat_id = cursor.lastrowid
    db.commit()
    db.close()
    return chat_id


def get_chats():
    db = db_connect()
    rows = db.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY id DESC"
    ).fetchall()
    db.close()
    return rows


def get_chat(chat_id):
    db = db_connect()
    row = db.execute(
        "SELECT id, title, created_at FROM conversations WHERE id=?",
        (chat_id,),
    ).fetchone()
    db.close()
    return row


def rename_chat(chat_id, title):
    title = title.strip()
    if not title:
        return

    db = db_connect()
    db.execute(
        "UPDATE conversations SET title=? WHERE id=?",
        (title, chat_id),
    )
    db.commit()
    db.close()


def delete_chat(chat_id):
    db = db_connect()
    db.execute("DELETE FROM attachments WHERE conversation_id=?", (chat_id,))
    db.execute("DELETE FROM messages WHERE conversation_id=?", (chat_id,))
    db.execute("DELETE FROM conversations WHERE id=?", (chat_id,))
    db.commit()
    db.close()


def save_message(chat_id, role, content):
    db = db_connect()
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO messages(conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (chat_id, role, content, datetime.now().isoformat()),
    )
    message_id = cursor.lastrowid
    db.commit()
    db.close()
    return message_id


def get_messages(chat_id):
    db = db_connect()
    rows = db.execute(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE conversation_id=?
        ORDER BY id
        """,
        (chat_id,),
    ).fetchall()
    db.close()

    return [
        {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
        for row in rows
    ]


# ================================================================
# MEMORY
# ================================================================

def get_memories():
    db = db_connect()
    rows = db.execute(
        "SELECT id, content, created_at FROM memories ORDER BY id DESC"
    ).fetchall()
    db.close()
    return rows


def add_memory(content):
    content = content.strip()
    if not content:
        return

    db = db_connect()
    existing = db.execute(
        "SELECT id FROM memories WHERE LOWER(content)=LOWER(?)",
        (content,),
    ).fetchone()

    if not existing:
        db.execute(
            "INSERT INTO memories(content, created_at) VALUES (?, ?)",
            (content, datetime.now().isoformat()),
        )

    db.commit()
    db.close()


def remove_memory(memory_id):
    db = db_connect()
    db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    db.commit()
    db.close()


def relevant_memories(query, limit=5):
    query_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))
    if not query_words:
        return []

    scores = []

    for memory_id, content, created in get_memories():
        words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", content.lower()))
        score = len(query_words & words)
        if score:
            scores.append((score, content))

    scores.sort(key=lambda x: x[0], reverse=True)

    return [content for score, content in scores[:limit]]


# ================================================================
# ATTACHMENTS
# ================================================================

def attachment_kind(path):
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return "pdf"

    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return "image"

    if suffix in (".txt", ".md", ".py", ".json", ".csv", ".html", ".css", ".js"):
        return "text"

    return "file"


def extract_pdf(path):
    if PdfReader is None:
        return "[PDF support unavailable. Install pypdf.]"

    try:
        reader = PdfReader(path)
        pieces = []

        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                pieces.append(f"\n--- PAGE {number} ---\n{text}")

            if sum(len(piece) for piece in pieces) >= MAX_PDF_CHARS:
                break

        result = "".join(pieces)

        if not result.strip():
            return "[No machine-readable text was found in this PDF.]"

        return result[:MAX_PDF_CHARS]

    except Exception as error:
        return f"[PDF reading error: {error}]"


def extract_text_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            return file.read(MAX_TEXT_ATTACHMENT_CHARS)
    except Exception as error:
        return f"[Text-file reading error: {error}]"


def image_to_base64(path):
    try:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("ascii")
    except Exception:
        return None


# ================================================================
# CONTEXT
# ================================================================

def build_context(chat_id, current_message, attachment_context=""):
    memories = relevant_memories(current_message)

    system = SYSTEM_PROMPT

    if memories:
        system += "\n\nRELEVANT LONG-TERM MEMORY:\n"
        for memory in memories:
            system += f"- {memory}\n"

    history = get_messages(chat_id)
    history = history[-MAX_HISTORY_MESSAGES:]

    context = [{"role": "system", "content": system}]

    for message in history:
        content = message["content"]

        # Current attachment context belongs
        # only to the newest user message.
        if (
            message["role"] == "user"
            and message is history[-1]
            and attachment_context
        ):
            content += "\n\nATTACHED DOCUMENT CONTENT:\n" + attachment_context

        context.append({"role": message["role"], "content": content})

    return context


# ================================================================
# MARKDOWN / RICH RENDERING
# ================================================================

def render_inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r'<code class="inline">\1</code>', text)
    return text


def markdown_to_html(text):
    """
    Lightweight safe renderer.

    It intentionally handles only the features GRAC needs:
    headings, bold, inline code, lists, formulas and code blocks.
    """

    lines = text.splitlines()
    output = []

    in_code = False
    code_lines = []
    code_language = ""

    in_formula = False
    formula_lines = []

    in_list = False

    for line in lines:
        stripped = line.strip()

        # ----------------------------------------------------------
        # CODE BLOCK
        # ----------------------------------------------------------
        if stripped.startswith("```"):
            if not in_code:
                if in_list:
                    output.append("</ul>")
                    in_list = False

                in_code = True
                code_language = stripped[3:].strip()
                code_lines = []
            else:
                code = html.escape("\n".join(code_lines))
                language = html.escape(code_language or "CODE")

                output.append(
                    f"""
                    <div class="codecard">
                        <div class="codetop">{language}</div>
                        <pre>{code}</pre>
                    </div>
                    """
                )

                in_code = False
                code_lines = []

            continue

        if in_code:
            code_lines.append(line)
            continue

        # ----------------------------------------------------------
        # FORMULA CARD
        # ----------------------------------------------------------
        if stripped == "$$":
            if not in_formula:
                if in_list:
                    output.append("</ul>")
                    in_list = False

                in_formula = True
                formula_lines = []
            else:
                formula = html.escape("\n".join(formula_lines).strip())
                output.append(f'<div class="formula">{formula}</div>')
                in_formula = False
                formula_lines = []

            continue

        if in_formula:
            formula_lines.append(line)
            continue

        # ----------------------------------------------------------
        # BLANK
        # ----------------------------------------------------------
        if not stripped:
            if in_list:
                output.append("</ul>")
                in_list = False

            output.append('<div class="space"></div>')
            continue

        # ----------------------------------------------------------
        # HEADINGS
        # ----------------------------------------------------------
        if stripped.startswith("### "):
            output.append("<h3>" + render_inline(stripped[4:]) + "</h3>")
            continue

        if stripped.startswith("## "):
            output.append("<h2>" + render_inline(stripped[3:]) + "</h2>")
            continue

        if stripped.startswith("# "):
            output.append("<h1>" + render_inline(stripped[2:]) + "</h1>")
            continue

        # ----------------------------------------------------------
        # BULLETS
        # ----------------------------------------------------------
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                output.append("<ul>")
                in_list = True

            output.append("<li>" + render_inline(stripped[2:]) + "</li>")
            continue

        if in_list:
            output.append("</ul>")
            in_list = False

        # ----------------------------------------------------------
        # NORMAL TEXT
        # ----------------------------------------------------------
        output.append("<p>" + render_inline(stripped) + "</p>")

    if in_list:
        output.append("</ul>")

    if in_code:
        code = html.escape("\n".join(code_lines))
        output.append(f"<pre>{code}</pre>")

    if in_formula:
        formula = html.escape("\n".join(formula_lines))
        output.append(f'<div class="formula">{formula}</div>')

    body = "\n".join(output)

    return f"""
    <html>
    <head>
    <style>

    body {{
        color: #E7E9ED;
        font-family: "Segoe UI";
        font-size: 14px;
        line-height: 1.55;
        background: transparent;
    }}

    p {{
        margin-top: 5px;
        margin-bottom: 8px;
    }}

    h1 {{
        color: #FFFFFF;
        font-size: 23px;
        margin-top: 16px;
        margin-bottom: 9px;
    }}

    h2 {{
        color: #FFFFFF;
        font-size: 19px;
        margin-top: 15px;
        margin-bottom: 8px;
    }}

    h3 {{
        color: #FFFFFF;
        font-size: 16px;
        margin-top: 12px;
        margin-bottom: 7px;
    }}

    strong {{
        color: #FFFFFF;
        font-weight: 700;
    }}

    ul {{
        margin-top: 5px;
        margin-bottom: 8px;
    }}

    li {{
        margin-bottom: 5px;
    }}

    code.inline {{
        background: #252A31;
        color: #F4F6F8;
        padding: 3px 6px;
        border-radius: 6px;
        font-family: Consolas;
    }}

    .formula {{
        background: #171B21;
        color: #FFFFFF;
        border: 1px solid #343A44;
        border-radius: 15px;
        padding: 17px 22px;
        margin-top: 12px;
        margin-bottom: 12px;
        font-family: "Cambria Math";
        font-size: 20px;
        font-weight: 600;
        text-align: center;
    }}

    .codecard {{
        background: #111419;
        border: 1px solid #303640;
        border-radius: 14px;
        margin-top: 11px;
        margin-bottom: 11px;
    }}

    .codetop {{
        background: #1B2027;
        color: #969DA8;
        padding: 8px 13px;
        font-size: 10px;
        font-weight: 700;
    }}

    pre {{
        color: #E5E7EB;
        font-family: Consolas;
        font-size: 12px;
        white-space: pre-wrap;
        padding: 13px;
        margin: 0;
    }}

    .space {{
        height: 5px;
    }}

    </style>
    </head>

    <body>
    {body}
    </body>
    </html>
    """


# ================================================================
# LOCAL IMAGE GENERATION — OPENVINO / INTEL GPU
# ================================================================

class ImageGenerationWorker(QThread):
    """
    Persistent OpenVINO image worker.

    The Stable Diffusion pipeline is loaded/compiled once, lazily, on the
    first image request. The same worker thread then stays alive and reuses
    that pipeline for later generations, avoiding the long GPU compile on
    every click.
    """
    status = Signal(str)
    completed = Signal(str, str)   # path, prompt
    failed = Signal(str, str)      # error, prompt

    def __init__(self):
        super().__init__()
        self.jobs = queue.Queue()
        self._shutdown_requested = False
        self.pipe = None

    def submit(self, prompt):
        self.jobs.put(prompt)

    def shutdown(self):
        self._shutdown_requested = True
        self.jobs.put(None)

    def _ensure_pipeline(self):
        if self.pipe is not None:
            return

        if openvino_genai is None or Image is None:
            raise RuntimeError(
                "Image packages are missing. Run:\n"
                "python -m pip install openvino-genai pillow"
            )

        if not IMAGE_MODEL_DIR.exists():
            raise RuntimeError(
                "GRAC cannot find the image model at:\n"
                f"{IMAGE_MODEL_DIR}\n\n"
                "Download the Stable Diffusion OpenVINO model into that folder."
            )

        self.status.emit(
            "Loading image AI on Intel GPU — first load can take a few minutes"
        )

        # Official OpenVINO GenAI API: model path + target device.
        self.pipe = openvino_genai.Text2ImagePipeline(
            str(IMAGE_MODEL_DIR),
            IMAGE_DEVICE
        )

        self.status.emit("Image AI loaded")

    def run(self):
        while not self._shutdown_requested:
            prompt = self.jobs.get()

            if prompt is None:
                break

            try:
                self._ensure_pipeline()

                self.status.emit("Creating image")

                seed = random.randint(0, 2_147_483_647)

                result = self.pipe.generate(
                    prompt,
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    num_inference_steps=IMAGE_STEPS,
                    num_images_per_prompt=1,
                    rng_seed=seed
                )

                self.status.emit("Saving image")

                image_array = result.data[0]
                image = Image.fromarray(image_array)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output = GENERATED_DIR / f"GRAC_{timestamp}.png"
                image.save(output)

                self.completed.emit(str(output), prompt)

            except Exception as error:
                self.failed.emit(str(error), prompt)


# ================================================================
# GENERATION WORKER
# ================================================================

class GenerationWorker(QThread):
    status = Signal(str)
    token = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, chat_id, user_text, model, attachment_context="", image_paths=None):
        super().__init__()

        self.chat_id = chat_id
        self.user_text = user_text
        self.model = model
        self.attachment_context = attachment_context
        self.image_paths = image_paths or []

        self.response = None
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

        if self.response:
            try:
                self.response.close()
            except Exception:
                pass

    def run(self):
        answer = ""

        try:
            self.status.emit("Understanding")

            messages = build_context(
                self.chat_id, self.user_text, self.attachment_context
            )

            selected_model = self.model

            # --------------------------------------------------------
            # IMAGE ROUTING
            # --------------------------------------------------------
            if self.image_paths:
                if not VISION_MODEL:
                    self.failed.emit(
                        "An image is attached, but GRAC does not "
                        "have a vision model configured yet."
                    )
                    return

                selected_model = VISION_MODEL

                images = []
                for path in self.image_paths:
                    encoded = image_to_base64(path)
                    if encoded:
                        images.append(encoded)

                if images:
                    messages[-1]["images"] = images

            self.status.emit("Preparing response")

            self.response = requests.post(
                OLLAMA_URL,
                json={
                    "model": selected_model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "keep_alive": KEEP_ALIVE,
                    "options": OLLAMA_OPTIONS,
                },
                stream=True,
                timeout=300,
            )

            try:
                self.response.raise_for_status()
            except requests.HTTPError as exc:
                if self.image_paths and self.response.status_code == 404:
                    self.failed.emit(
                        f"GRAC's vision model ({VISION_MODEL}) is not installed in Ollama yet.\n\n"
                        f"Run this once in PowerShell:\nollama pull {VISION_MODEL}"
                    )
                    return
                raise exc

            first = True

            for line in self.response.iter_lines():
                if self.stop_requested:
                    break

                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                text = data.get("message", {}).get("content", "")

                if text:
                    if first:
                        first = False
                        self.status.emit("Generating")

                    answer += text
                    self.token.emit(text)

                if data.get("done", False):
                    break

            if answer:
                save_message(self.chat_id, "assistant", answer)

            self.completed.emit(answer)

        except Exception as error:
            if not self.stop_requested:
                self.failed.emit(str(error))


# ================================================================
# TITLE WORKER
# ================================================================

class TitleWorker(QThread):
    ready = Signal(int, str)

    def __init__(self, chat_id, text, model):
        super().__init__()
        self.chat_id = chat_id
        self.text = text
        self.model = model

    def run(self):
        fallback = " ".join(self.text.split()[:5])[:45]

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Create a 2 to 5 word title "
                                "for this conversation. "
                                "Return ONLY the title.\n\n" + self.text
                            ),
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": KEEP_ALIVE,
                    "options": {"num_ctx": 1024},
                },
                timeout=60,
            )

            response.raise_for_status()

            title = (
                response.json()["message"]["content"]
                .strip()
                .splitlines()[0]
                .strip("\"'")
            )[:45]

        except Exception:
            title = fallback

        if not title:
            title = "New chat"

        rename_chat(self.chat_id, title)
        self.ready.emit(self.chat_id, title)


# ================================================================
# MEMORY WORKER
# ================================================================

class MemoryWorker(QThread):
    def __init__(self, text, model):
        super().__init__()
        self.text = text
        self.model = model

    def run(self):
        prompt = f"""

Extract only durable information that could be useful
to this user's personal AI in future conversations.

Store things such as:

preferences
long-running projects
goals
stable instructions

Do NOT store:

passwords
API keys
authentication secrets
greetings
temporary statements

Return JSON exactly in this structure:

{{"memories":[]}}

USER MESSAGE:
{self.text}
"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "keep_alive": KEEP_ALIVE,
                    "options": {"num_ctx": 1536},
                },
                timeout=90,
            )

            response.raise_for_status()

            result = json.loads(response.json()["message"]["content"])
            memories = result.get("memories", [])

            if isinstance(memories, list):
                for memory in memories:
                    if isinstance(memory, str):
                        add_memory(memory)

        except Exception:
            pass


# ================================================================
# CHAT SIDEBAR ITEM
# ================================================================

class ChatItem(QFrame):
    clicked = Signal(int)
    rename_requested = Signal(int)
    delete_requested = Signal(int)

    def __init__(self, chat_id, title):
        super().__init__()

        self.chat_id = chat_id

        self.setObjectName("chatItem")
        self.setFixedHeight(42)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 5, 0)

        self.label = QLabel(title)
        self.label.setObjectName("chatTitle")

        self.more = QPushButton("•••")
        self.more.setObjectName("moreButton")
        self.more.setFixedSize(30, 30)
        self.more.clicked.connect(self.menu)

        layout.addWidget(self.label, 1)
        layout.addWidget(self.more)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.chat_id)
        elif event.button() == Qt.RightButton:
            self.menu()

    def menu(self):
        menu = QMenu(self)

        rename = menu.addAction("Rename")
        delete = menu.addAction("Delete")

        rename.triggered.connect(lambda: self.rename_requested.emit(self.chat_id))
        delete.triggered.connect(lambda: self.delete_requested.emit(self.chat_id))

        menu.exec(self.more.mapToGlobal(self.more.rect().bottomLeft()))


# ================================================================
# ATTACHMENT CHIP
# ================================================================

class AttachmentChip(QFrame):
    remove_requested = Signal(str)

    def __init__(self, path):
        super().__init__()

        self.path = path
        self.setObjectName("attachmentChip")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 7, 7)

        kind = attachment_kind(path)

        icons = {"pdf": "PDF", "image": "IMG", "text": "TXT", "file": "FILE"}
        icon = QLabel(icons.get(kind, "FILE"))
        icon.setObjectName("attachmentType")

        filename = QLabel(Path(path).name)
        filename.setObjectName("attachmentName")

        close = QPushButton("×")
        close.setObjectName("attachmentClose")
        close.setFixedSize(24, 24)
        close.clicked.connect(lambda: self.remove_requested.emit(self.path))

        layout.addWidget(icon)
        layout.addWidget(filename)
        layout.addWidget(close)


# ================================================================
# GENERATED IMAGE WIDGET
# ================================================================

class GeneratedImageWidget(QFrame):
    def __init__(self, image_path, prompt="Generated by GRAC"):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.setObjectName("generatedImageCard")
        self.setMaximumWidth(700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top = QHBoxLayout()

        icon = QLabel("G")
        icon.setObjectName("assistantIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(29, 29)

        title = QLabel("GRAC · IMAGE")
        title.setObjectName("assistantName")

        top.addWidget(icon)
        top.addWidget(title)
        top.addStretch()
        layout.addLayout(top)

        preview = QLabel()
        preview.setAlignment(Qt.AlignCenter)
        preview.setObjectName("generatedPreview")

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            preview.setPixmap(
                pixmap.scaled(
                    620, 620,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        else:
            preview.setText("Unable to display image.")

        layout.addWidget(preview)

        prompt_label = QLabel(prompt)
        prompt_label.setWordWrap(True)
        prompt_label.setObjectName("generatedPrompt")
        layout.addWidget(prompt_label)

        buttons = QHBoxLayout()

        open_button = QPushButton("Open image")
        open_button.setObjectName("miniButton")
        open_button.clicked.connect(self.open_image)

        copy_button = QPushButton("Copy path")
        copy_button.setObjectName("miniButton")
        copy_button.clicked.connect(self.copy_path)

        buttons.addWidget(open_button)
        buttons.addWidget(copy_button)
        buttons.addStretch()
        layout.addLayout(buttons)

    def open_image(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))

    def copy_path(self):
        QApplication.clipboard().setText(self.image_path)


# ================================================================
# MESSAGE WIDGET
# ================================================================

class MessageWidget(QWidget):
    def __init__(self, role, content=""):
        super().__init__()

        self.role = role
        self.raw_content = content

        outer = QHBoxLayout(self)
        outer.setContentsMargins(28, 10, 28, 10)

        if role == "user":
            outer.addStretch()

            bubble = QFrame()
            bubble.setObjectName("userBubble")
            bubble.setMaximumWidth(720)

            bubble_layout = QVBoxLayout(bubble)
            bubble_layout.setContentsMargins(16, 11, 16, 11)

            self.text = QLabel(content)
            self.text.setObjectName("userMessage")
            self.text.setWordWrap(True)
            self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)

            bubble_layout.addWidget(self.text)
            outer.addWidget(bubble)

        else:
            card = QWidget()
            card.setMaximumWidth(880)

            layout = QVBoxLayout(card)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)

            top = QHBoxLayout()

            icon = QLabel("G")
            icon.setObjectName("assistantIcon")
            icon.setAlignment(Qt.AlignCenter)
            icon.setFixedSize(29, 29)

            name = QLabel("GRAC")
            name.setObjectName("assistantName")

            copy_button = QPushButton("Copy")
            copy_button.setObjectName("miniButton")
            copy_button.clicked.connect(self.copy_response)

            top.addWidget(icon)
            top.addWidget(name)
            top.addStretch()
            top.addWidget(copy_button)

            layout.addLayout(top)

            self.text = QTextBrowser()
            self.text.setObjectName("assistantMessage")
            self.text.setFrameShape(QFrame.NoFrame)
            self.text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.text.setOpenExternalLinks(True)
            self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.text.document().documentLayout().documentSizeChanged.connect(self._fit_document)

            layout.addWidget(self.text)

            outer.addWidget(card, 1)
            outer.addStretch()

            self.set_content(content)

    def _fit_document(self, *_):
        if self.role != "user":
            h = int(self.text.document().documentLayout().documentSize().height()) + 18
            self.text.setMinimumHeight(max(48, h))
            self.text.setMaximumHeight(max(48, h))

    def copy_response(self):
        QApplication.clipboard().setText(self.raw_content)

    def set_content(self, content):
        self.raw_content = content

        if self.role == "user":
            self.text.setText(content)
            return

        self.text.setHtml(markdown_to_html(content))

        self._fit_document()


# ================================================================
# MEMORY DIALOG
# ================================================================

class MemoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("GRAC Memory")
        self.resize(650, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Long-term memory")
        title.setObjectName("dialogHeading")

        description = QLabel("Information GRAC can recall across conversations.")
        description.setObjectName("dialogSubtitle")

        self.list = QListWidget()
        self.list.setObjectName("memoryList")

        buttons = QHBoxLayout()

        add = QPushButton("+ Add memory")
        delete = QPushButton("Delete selected")

        add.clicked.connect(self.add_item)
        delete.clicked.connect(self.delete_item)

        buttons.addWidget(add)
        buttons.addStretch()
        buttons.addWidget(delete)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(10)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)

        self.reload()

    def reload(self):
        self.list.clear()

        for memory_id, content, created in get_memories():
            item = QListWidgetItem(content)
            item.setData(Qt.UserRole, memory_id)
            self.list.addItem(item)

    def add_item(self):
        text, ok = QInputDialog.getMultiLineText(
            self, "Add memory", "What should GRAC remember?"
        )

        if ok and text.strip():
            add_memory(text)
            self.reload()

    def delete_item(self):
        item = self.list.currentItem()
        if not item:
            return

        remove_memory(item.data(Qt.UserRole))
        self.reload()


# ================================================================
# MAIN WINDOW
# ================================================================

class GRACWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_chat = None
        self.generating = False

        self.worker = None
        self.title_worker = None
        self.memory_worker = None
        self.image_worker = None

        self.first_message = False

        self.ai_widget = None
        self.ai_text = ""

        self.pending_attachments = []

        self.loading_status = ""
        self.loading_dots = 0

        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.animate_loading)

        self.setWindowTitle("GRAC")
        self.resize(1350, 850)
        self.setMinimumSize(950, 650)

        self.build_ui()
        self.apply_style()
        self.load_chats()
        self.show_welcome()
        self.install_shortcuts()

    # ============================================================
    # BUILD UI
    # ============================================================

    def build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ========================================================
        # SIDEBAR
        # ========================================================

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(270)

        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(12, 15, 12, 14)
        side.setSpacing(8)

        brand_row = QHBoxLayout()

        logo = QLabel("G")
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(35, 35)

        brand = QLabel("GRAC")
        brand.setObjectName("brand")

        brand_row.addWidget(logo)
        brand_row.addWidget(brand)
        brand_row.addStretch()

        side.addLayout(brand_row)
        side.addSpacing(8)

        new_chat = QPushButton("＋   New chat")
        new_chat.setObjectName("newChatButton")
        new_chat.clicked.connect(self.new_chat)
        side.addWidget(new_chat)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search conversations")
        self.search.setObjectName("search")
        self.search.textChanged.connect(self.load_chats)
        side.addWidget(self.search)

        section = QLabel("CONVERSATIONS")
        section.setObjectName("sectionHeading")
        side.addWidget(section)

        # Full conversation list: no tiny nested chat scroll box.
        self.chat_container = QWidget()
        self.chat_container.setObjectName("chatContainer")
        self.chat_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(3)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.addStretch()

        side.addWidget(self.chat_container, 1)

        memory = QPushButton("◇   Memory")
        memory.setObjectName("sideButton")
        memory.clicked.connect(self.open_memory)
        side.addWidget(memory)

        status = QLabel("●  Ollama · Local")
        status.setObjectName("status")
        side.addWidget(status)

        root_layout.addWidget(self.sidebar)

        # ========================================================
        # MAIN
        # ========================================================

        main = QFrame()
        main.setObjectName("main")

        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # TOPBAR

        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(65)

        top = QHBoxLayout(topbar)
        top.setContentsMargins(20, 0, 22, 0)

        menu = QPushButton("☰")
        menu.setObjectName("iconButton")
        menu.setFixedSize(35, 35)
        menu.clicked.connect(self.toggle_sidebar)

        self.title = QLabel("GRAC")
        self.title.setObjectName("headerTitle")

        self.model_selector = QComboBox()
        self.model_selector.setObjectName("modelSelector")
        self.model_selector.addItems([DEFAULT_MODEL, "qwen3:1.7b"])

        local = QLabel("● PRIVATE · LOCAL")
        local.setObjectName("localPill")

        top.addWidget(menu)
        top.addSpacing(6)
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.model_selector)
        top.addSpacing(8)
        top.addWidget(local)

        main_layout.addWidget(topbar)

        # CONVERSATION

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setObjectName("conversationScroll")

        self.page = QWidget()
        self.page.setObjectName("conversationPage")

        self.messages_layout = QVBoxLayout(self.page)
        self.messages_layout.setContentsMargins(50, 25, 50, 35)
        self.messages_layout.setSpacing(4)

        self.scroll.setWidget(self.page)
        main_layout.addWidget(self.scroll, 1)

        # ========================================================
        # COMPOSER
        # ========================================================

        composer_area = QFrame()
        composer_area.setObjectName("composerArea")

        area = QVBoxLayout(composer_area)
        area.setContentsMargins(55, 8, 55, 17)

        self.attachment_row = QWidget()
        self.attachment_layout = QHBoxLayout(self.attachment_row)
        self.attachment_layout.setContentsMargins(0, 0, 0, 0)
        self.attachment_layout.addStretch()

        area.addWidget(self.attachment_row)

        self.composer = QFrame()
        self.composer.setObjectName("composer")
        self.composer.setMaximumWidth(950)

        composer_layout = QVBoxLayout(self.composer)
        composer_layout.setContentsMargins(10, 8, 10, 8)

        self.input = QTextEdit()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("Ask GRAC anything...")
        self.input.setAcceptRichText(False)
        self.input.setFixedHeight(50)
        self.input.textChanged.connect(self.resize_input)

        composer_layout.addWidget(self.input)

        bottom = QHBoxLayout()

        self.attach_button = QPushButton("+")
        self.attach_button.setObjectName("composerTool")
        self.attach_button.setFixedSize(34, 34)
        self.attach_button.clicked.connect(self.attachment_menu)

        image_button = QPushButton("Image")
        image_button.setObjectName("composerToolText")
        image_button.clicked.connect(self.image_generation_notice)

        video_button = QPushButton("Video")
        video_button.setObjectName("composerToolText")
        video_button.clicked.connect(self.video_generation_notice)

        bottom.addWidget(self.attach_button)
        bottom.addWidget(image_button)
        bottom.addWidget(video_button)
        bottom.addStretch()

        self.send_button = QPushButton("↑")
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedSize(42, 42)
        self.send_button.clicked.connect(self.send_or_stop)

        bottom.addWidget(self.send_button)

        composer_layout.addLayout(bottom)

        centered = QHBoxLayout()
        centered.addStretch()
        centered.addWidget(self.composer, 1)
        centered.addStretch()

        area.addLayout(centered)

        footer = QLabel("GRAC can make mistakes · Local AI")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)

        area.addWidget(footer)

        main_layout.addWidget(composer_area)
        root_layout.addWidget(main, 1)

    # ============================================================
    # SHORTCUTS
    # ============================================================

    def install_shortcuts(self):
        new = QShortcut(QKeySequence("Ctrl+N"), self)
        new.activated.connect(self.new_chat)

        search = QShortcut(QKeySequence("Ctrl+K"), self)
        search.activated.connect(self.focus_search)

        self.new_shortcut = new
        self.search_shortcut = search

    def focus_search(self):
        self.sidebar.show()
        self.search.setFocus()

    # ============================================================
    # KEYBOARD
    # ============================================================

    def keyPressEvent(self, event):
        if (
            self.input.hasFocus()
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and not (event.modifiers() & Qt.ShiftModifier)
        ):
            self.send_or_stop()
            return

        super().keyPressEvent(event)

    # ============================================================
    # INPUT RESIZE
    # ============================================================

    def resize_input(self):
        height = int(self.input.document().size().height()) + 18
        self.input.setFixedHeight(max(50, min(height, 160)))

    # ============================================================
    # SIDEBAR
    # ============================================================

    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def clear_chat_list(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def load_chats(self):
        self.clear_chat_list()

        query = self.search.text().lower().strip()

        for chat_id, title, created in get_chats():
            if query and query not in title.lower():
                continue

            item = ChatItem(chat_id, title)
            item.clicked.connect(self.open_chat)
            item.rename_requested.connect(self.rename_dialog)
            item.delete_requested.connect(self.delete_dialog)

            self.chat_layout.insertWidget(self.chat_layout.count() - 1, item)

    # ============================================================
    # MESSAGES
    # ============================================================

    def clear_messages(self):
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_message(self, role, content):
        count = self.messages_layout.count()

        if count:
            last = self.messages_layout.itemAt(count - 1)
            if last.spacerItem():
                self.messages_layout.takeAt(count - 1)

        widget = MessageWidget(role, content)

        self.messages_layout.addWidget(widget)
        self.messages_layout.addStretch()

        self.scroll_bottom()

        return widget

    def add_generated_image(self, image_path, prompt="Generated by GRAC"):
        count = self.messages_layout.count()

        if count:
            last = self.messages_layout.itemAt(count - 1)
            if last.spacerItem():
                self.messages_layout.takeAt(count - 1)

        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(28, 10, 28, 10)

        card = GeneratedImageWidget(image_path, prompt)
        row.addWidget(card)
        row.addStretch()

        self.messages_layout.addWidget(container)
        self.messages_layout.addStretch()
        self.scroll_bottom()

    # ============================================================
    # WELCOME
    # ============================================================

    def show_welcome(self):
        self.clear_messages()
        self.current_chat = None
        self.title.setText("GRAC")

        self.messages_layout.addStretch(2)

        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setAlignment(Qt.AlignCenter)

        logo = QLabel("G")
        logo.setObjectName("welcomeLogo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(70, 70)

        title = QLabel("GRAC")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("How can I help?")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        description = QLabel("Chat · Documents · Images · Memory")
        description.setObjectName("welcomeDescription")
        description.setAlignment(Qt.AlignCenter)

        layout.addWidget(logo, 0, Qt.AlignCenter)
        layout.addSpacing(15)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(description)

        self.messages_layout.addWidget(center)
        self.messages_layout.addStretch(3)

        self.input.setFocus()

    def new_chat(self):
        if self.generating:
            return

        self.pending_attachments = []
        self.refresh_attachments()
        self.show_welcome()

    # ============================================================
    # OPEN CHAT
    # ============================================================

    def open_chat(self, chat_id):
        if self.generating:
            return

        chat = get_chat(chat_id)
        if not chat:
            return

        self.current_chat = chat_id
        self.title.setText(chat[1])

        self.clear_messages()

        for message in get_messages(chat_id):
            content = message["content"]

            if (
                message["role"] == "assistant"
                and content.startswith("[Generated image]\n")
            ):
                lines = content.splitlines()
                image_path = lines[1].strip() if len(lines) > 1 else ""
                prompt = "Generated by GRAC"

                for line in lines[2:]:
                    if line.startswith("PROMPT: "):
                        prompt = line[8:].strip()
                        break

                if image_path and Path(image_path).exists():
                    self.add_generated_image(image_path, prompt)
                else:
                    self.add_message(
                        "assistant",
                        "Generated image file is missing:\n" + image_path
                    )
            else:
                self.add_message(message["role"], content)

        self.scroll_bottom()

    # ============================================================
    # ATTACHMENTS
    # ============================================================

    def attachment_menu(self):
        menu = QMenu(self)

        image = menu.addAction("Attach photo")
        pdf = menu.addAction("Attach PDF")
        text = menu.addAction("Attach text/code file")

        image.triggered.connect(self.attach_image)
        pdf.triggered.connect(self.attach_pdf)
        text.triggered.connect(self.attach_text)

        menu.exec(self.attach_button.mapToGlobal(self.attach_button.rect().topLeft()))

    def attach_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        self.add_attachment_paths(paths)

    def attach_pdf(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach PDFs", "", "PDF files (*.pdf)"
        )
        self.add_attachment_paths(paths)

    def attach_text(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach files",
            "",
            "Text and code (*.txt *.md *.py *.json *.csv *.html *.css *.js)",
        )
        self.add_attachment_paths(paths)

    def add_attachment_paths(self, paths):
        for path in paths:
            if path not in self.pending_attachments:
                self.pending_attachments.append(path)

        self.refresh_attachments()

    def remove_attachment(self, path):
        if path in self.pending_attachments:
            self.pending_attachments.remove(path)

        self.refresh_attachments()

    def refresh_attachments(self):
        while self.attachment_layout.count() > 1:
            item = self.attachment_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for path in self.pending_attachments:
            chip = AttachmentChip(path)
            chip.remove_requested.connect(self.remove_attachment)
            self.attachment_layout.insertWidget(self.attachment_layout.count() - 1, chip)

    # ============================================================
    # GENERATION BUTTONS
    # ============================================================

    def image_generation_notice(self):
        if self.generating:
            QMessageBox.information(
                self, "GRAC is busy",
                "Wait for the current task to finish first."
            )
            return

        prompt, ok = QInputDialog.getMultiLineText(
            self,
            "Generate image",
            "Describe the image you want GRAC to create:"
        )

        if not ok or not prompt.strip():
            return

        prompt = prompt.strip()

        if self.current_chat is None:
            self.current_chat = create_chat()
            self.first_message = True
            self.clear_messages()
            self.title.setText("Image generation")
        else:
            self.first_message = False

        user_text = f"Generate image: {prompt}"
        save_message(self.current_chat, "user", user_text)
        self.add_message("user", user_text)

        self.ai_text = ""
        self.ai_widget = self.add_message("assistant", "Preparing image AI......")

        self.generating = True
        self.send_button.setText("■")

        # Keep one long-lived image worker so OpenVINO compiles Stable
        # Diffusion only once per GRAC session.
        if self.image_worker is None:
            self.image_worker = ImageGenerationWorker()
            self.image_worker.status.connect(self.image_generation_status)
            self.image_worker.completed.connect(self.image_generation_finished)
            self.image_worker.failed.connect(self.image_generation_error)
            self.image_worker.start()

        self.image_worker.submit(prompt)

        self.load_chats()

    def image_generation_status(self, status):
        if self.ai_widget:
            self.ai_widget.set_content(status + "......")

    def image_generation_finished(self, path, prompt):
        self.generating = False
        self.send_button.setText("↑")

        if self.ai_widget:
            self.ai_widget.deleteLater()
            self.ai_widget = None

        save_message(
            self.current_chat,
            "assistant",
            f"[Generated image]\n{path}\nPROMPT: {prompt}"
        )

        self.add_generated_image(path, prompt)

        if self.first_message:
            fallback = "Image · " + " ".join(prompt.split()[:4])
            rename_chat(self.current_chat, fallback[:45])
            self.title.setText(fallback[:45])

        self.load_chats()
        self.input.setFocus()

    def image_generation_error(self, error, prompt=""):
        self.generating = False
        self.send_button.setText("↑")

        if self.ai_widget:
            self.ai_widget.set_content(
                "## Image generation failed\n\n" + error
            )

    def video_generation_notice(self):
        QMessageBox.information(
            self,
            "Video generation",
            (
                "The GRAC video-generation interface is ready, "
                "but no video generator is configured yet.\n\n"
                "Video generation requires a separate model/backend "
                "and considerably more compute than the chat model."
            ),
        )

    # ============================================================
    # SEND
    # ============================================================

    def send_or_stop(self):
        if self.generating:
            self.stop_generation()
        else:
            self.send_message()

    def send_message(self):
        text = self.input.toPlainText().strip()

        if not text and not self.pending_attachments:
            return

        if not text:
            text = "Please examine the attached file."

        if self.current_chat is None:
            self.current_chat = create_chat()
            self.first_message = True
            self.clear_messages()
            self.title.setText("New chat")
        else:
            self.first_message = False

        # --------------------------------------------------------
        # ATTACHMENT CONTEXT
        # --------------------------------------------------------

        document_sections = []
        image_paths = []
        attachment_names = []

        for path in self.pending_attachments:
            kind = attachment_kind(path)
            attachment_names.append(Path(path).name)

            if kind == "pdf":
                document_sections.append(
                    f"\n\nFILE: {Path(path).name}\n" + extract_pdf(path)
                )
            elif kind == "text":
                document_sections.append(
                    f"\n\nFILE: {Path(path).name}\n" + extract_text_file(path)
                )
            elif kind == "image":
                image_paths.append(path)

        attachment_context = "".join(document_sections)

        display_text = text

        if attachment_names:
            display_text += "\n\nAttached: " + ", ".join(attachment_names)

        save_message(self.current_chat, "user", display_text)
        self.add_message("user", display_text)

        self.input.clear()

        attachments = list(self.pending_attachments)
        self.pending_attachments = []
        self.refresh_attachments()

        self.ai_text = ""
        self.ai_widget = self.add_message("assistant", "")

        self.generating = True
        self.send_button.setText("■")

        self.loading_status = "Understanding"
        self.loading_dots = 0

        self.loading_timer.start(230)
        self.animate_loading()

        model = self.model_selector.currentText()

        self.worker = GenerationWorker(
            self.current_chat, text, model, attachment_context, image_paths
        )

        self.worker.status.connect(self.update_status)
        self.worker.token.connect(self.receive_token)
        self.worker.completed.connect(
            lambda response: self.generation_finished(text, model)
        )
        self.worker.failed.connect(self.generation_error)

        self.worker.start()

        self.load_chats()

    # ============================================================
    # LOADING
    # ============================================================

    def update_status(self, status):
        if not self.generating:
            return

        self.loading_status = status
        self.loading_dots = 0
        self.animate_loading()

    def animate_loading(self):
        if not self.generating or self.ai_text or not self.ai_widget:
            return

        self.loading_dots += 1

        if self.loading_dots > 6:
            self.loading_dots = 1

        self.ai_widget.set_content(self.loading_status + ("." * self.loading_dots))

    def stop_loading(self):
        self.loading_timer.stop()
        self.loading_status = ""
        self.loading_dots = 0

    # ============================================================
    # STREAMING
    # ============================================================

    def receive_token(self, token):
        if not self.ai_text:
            self.stop_loading()

        self.ai_text += token

        if self.ai_widget:
            self.ai_widget.set_content(self.ai_text)

        self.scroll_bottom()

    # ============================================================
    # FINISHED
    # ============================================================

    def generation_finished(self, original_text, model):
        self.generating = False
        self.stop_loading()
        self.send_button.setText("↑")

        if self.first_message:
            self.title_worker = TitleWorker(self.current_chat, original_text, model)
            self.title_worker.ready.connect(
                lambda chat_id, title: self.title_finished(
                    chat_id, title, original_text, model
                )
            )
            self.title_worker.start()
        else:
            self.start_memory(original_text, model)

        self.input.setFocus()

    def title_finished(self, chat_id, title, original_text, model):
        if self.current_chat == chat_id:
            self.title.setText(title)

        self.load_chats()
        self.start_memory(original_text, model)

    def start_memory(self, text, model):
        if self.memory_worker and self.memory_worker.isRunning():
            return

        self.memory_worker = MemoryWorker(text, model)
        self.memory_worker.start()

    # ============================================================
    # ERROR / STOP
    # ============================================================

    def generation_error(self, error):
        self.generating = False
        self.stop_loading()
        self.send_button.setText("↑")

        if self.ai_widget:
            self.ai_widget.set_content(
                "## GRAC couldn't complete the request\n\n" + error
            )

    def stop_generation(self):
        if self.image_worker and self.image_worker.isRunning():
            QMessageBox.information(
                self,
                "Image generation",
                "The current OpenVINO image pass is already running. "
                "GRAC will finish this image before accepting another task."
            )
            return

        if self.worker:
            self.worker.stop()

        self.generating = False
        self.stop_loading()
        self.send_button.setText("↑")

        if self.ai_widget and not self.ai_text:
            self.ai_widget.set_content("Generation stopped.")

    # ============================================================
    # CHAT MANAGEMENT
    # ============================================================

    def rename_dialog(self, chat_id):
        chat = get_chat(chat_id)
        if not chat:
            return

        title, ok = QInputDialog.getText(
            self, "Rename conversation", "Conversation name:", text=chat[1]
        )

        if ok and title.strip():
            rename_chat(chat_id, title)

            if self.current_chat == chat_id:
                self.title.setText(title)

            self.load_chats()

    def delete_dialog(self, chat_id):
        chat = get_chat(chat_id)
        if not chat:
            return

        answer = QMessageBox.question(
            self,
            "Delete conversation",
            f'Delete "{chat[1]}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        delete_chat(chat_id)

        if self.current_chat == chat_id:
            self.show_welcome()

        self.load_chats()

    # ============================================================
    # MEMORY
    # ============================================================

    def open_memory(self):
        MemoryDialog(self).exec()

    # ============================================================
    # SCROLL
    # ============================================================

    def scroll_bottom(self):
        QTimer.singleShot(15, self._scroll_bottom)

    def _scroll_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ============================================================
    # STYLE
    # ============================================================


    def closeEvent(self, event):
        # Shut down the persistent image worker cleanly when GRAC closes.
        if self.image_worker is not None and self.image_worker.isRunning():
            self.image_worker.shutdown()
            self.image_worker.wait(3000)

        super().closeEvent(event)

    def apply_style(self):
        self.setStyleSheet("""
        * {
            font-family: "Segoe UI";
        }

        #root,
        #main,
        #conversationPage,
        #composerArea {
            background: #0A0C0F;
            color: #F5F6F8;
        }

        #sidebar {
            background: #0F1115;
            border-right: 1px solid #252930;
        }

        #logo,
        #welcomeLogo,
        #assistantIcon {
            background: #F5F6F8;
            color: #0A0C0F;
            font-weight: 900;
        }

        #logo {
            border-radius: 11px;
            font-size: 15px;
        }

        #assistantIcon {
            border-radius: 9px;
        }

        #welcomeLogo {
            border-radius: 23px;
            font-size: 29px;
        }

        #brand {
            color: #FFFFFF;
            font-size: 18px;
            font-weight: 750;
        }

        #newChatButton,
        #sideButton {
            background: transparent;
            color: #E5E7EB;
            border: none;
            border-radius: 10px;
            text-align: left;
            padding: 11px 12px;
            font-size: 13px;
        }

        #newChatButton {
            background: #191C21;
            border: 1px solid #2A2F37;
            font-weight: 650;
        }

        #newChatButton:hover,
        #sideButton:hover {
            background: #23272E;
        }

        #search {
            background: #171A1F;
            color: #F5F6F8;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 9px 11px;
        }

        #search:focus {
            border: 1px solid #404751;
        }

        #sectionHeading {
            color: #717884;
            font-size: 10px;
            font-weight: 700;
            padding: 8px;
        }

        #chatScroll,
        #chatContainer {
            border: none;
            background: transparent;
        }

        #chatItem {
            background: transparent;
            border-radius: 9px;
        }

        #chatItem:hover {
            background: #20242B;
        }

        #chatTitle {
            color: #D6DAE0;
            font-size: 13px;
        }

        #moreButton {
            background: transparent;
            color: #777F8B;
            border: none;
            border-radius: 7px;
        }

        #moreButton:hover {
            background: #30353D;
            color: white;
        }

        #status {
            color: #64D995;
            font-size: 10px;
            padding: 7px;
        }

        #topbar {
            background: #0A0C0F;
            border-bottom: 1px solid #242830;
        }

        #iconButton {
            background: transparent;
            color: #9299A4;
            border: none;
            border-radius: 9px;
            font-size: 17px;
        }

        #iconButton:hover {
            background: #20242B;
            color: white;
        }

        #headerTitle {
            color: #F5F6F8;
            font-size: 13px;
            font-weight: 700;
        }

        #modelSelector {
            background: #171A1F;
            color: #DDE1E6;
            border: 1px solid #2D323A;
            border-radius: 9px;
            padding: 6px 10px;
        }

        #localPill {
            color: #8D949E;
            background: #171A1F;
            border: 1px solid #2D323A;
            border-radius: 13px;
            padding: 6px 10px;
            font-size: 9px;
            font-weight: 700;
        }

        #conversationScroll {
            background: #0A0C0F;
            border: none;
        }

        #welcomeTitle {
            color: white;
            font-size: 32px;
            font-weight: 800;
        }

        #welcomeSubtitle {
            color: #E1E4E8;
            font-size: 20px;
        }

        #welcomeDescription {
            color: #717884;
            font-size: 11px;
        }

        #userBubble {
            background: #1B1F25;
            border: 1px solid #2D323A;
            border-radius: 18px;
        }

        #userMessage {
            color: #F5F6F8;
            font-size: 14px;
        }

        #assistantName {
            color: white;
            font-size: 12px;
            font-weight: 750;
        }

        #assistantMessage {
            background: transparent;
            border: none;
        }

        #miniButton {
            background: transparent;
            color: #747B86;
            border: none;
            border-radius: 7px;
            padding: 5px 8px;
        }

        #miniButton:hover {
            background: #20242B;
            color: white;
        }

        #composer {
            background: #171A1F;
            border: 1px solid #363C46;
            border-radius: 21px;
        }

        #composer:focus-within {
            border: 1px solid #484F5A;
        }

        #input {
            background: transparent;
            color: #F5F6F8;
            border: none;
            padding: 6px 8px;
            font-size: 14px;
        }

        #composerTool {
            background: transparent;
            color: #D9DDE2;
            border: none;
            border-radius: 10px;
            font-size: 22px;
        }

        #composerTool:hover {
            background: #292E36;
        }

        #composerToolText {
            background: transparent;
            color: #A5ABB5;
            border: none;
            border-radius: 9px;
            padding: 7px 9px;
            font-size: 11px;
        }

        #composerToolText:hover {
            background: #292E36;
            color: white;
        }

        #sendButton {
            background: #F5F6F8;
            color: #0A0C0F;
            border: none;
            border-radius: 14px;
            font-size: 19px;
            font-weight: 900;
        }

        #sendButton:hover {
            background: #DDE1E6;
        }

        #attachmentChip {
            background: #171A1F;
            border: 1px solid #303640;
            border-radius: 12px;
        }

        #attachmentType {
            background: #272C34;
            color: #AAB1BB;
            border-radius: 6px;
            padding: 4px 6px;
            font-size: 8px;
            font-weight: 800;
        }

        #attachmentName {
            color: #DCE0E5;
            font-size: 11px;
        }

        #attachmentClose {
            background: transparent;
            color: #8C939E;
            border: none;
            border-radius: 7px;
            font-size: 16px;
        }

        #attachmentClose:hover {
            background: #303640;
            color: white;
        }

        #footer {
            color: #555C66;
            font-size: 9px;
        }

        #generatedImageCard {
            background: #111419;
            border: 1px solid #303640;
            border-radius: 18px;
        }

        #generatedPreview {
            background: #0D0F13;
            border: 1px solid #252A31;
            border-radius: 14px;
            padding: 6px;
        }

        #generatedPrompt {
            color: #9DA4AE;
            font-size: 11px;
            padding: 3px 5px;
        }

        QDialog {
            background: #0D0F13;
            color: white;
        }

        #dialogHeading {
            color: white;
            font-size: 22px;
            font-weight: 750;
        }

        #dialogSubtitle {
            color: #858C97;
        }

        #memoryList {
            background: #171A1F;
            color: #E2E5E9;
            border: 1px solid #303640;
            border-radius: 12px;
            padding: 7px;
        }

        QPushButton {
            background: #20242B;
            color: #E8EAED;
            border: 1px solid #303640;
            border-radius: 9px;
            padding: 8px 12px;
        }

        QPushButton:hover {
            background: #292E36;
        }

        QMenu {
            background: #1B1F25;
            color: #F5F6F8;
            border: 1px solid #343A44;
            padding: 6px;
        }

        QMenu::item {
            padding: 8px 28px 8px 12px;
            border-radius: 6px;
        }

        QMenu::item:selected {
            background: #292E36;
        }

        QScrollBar:vertical {
            background: transparent;
            width: 8px;
        }

        QScrollBar::handle:vertical {
            background: #353B44;
            min-height: 35px;
            border-radius: 4px;
        }

        QScrollBar::handle:vertical:hover {
            background: #4A515C;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: transparent;
        }
        """)


# ================================================================
# START
# ================================================================

def main():
    setup_database()

    app = QApplication(sys.argv)

    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    window = GRACWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
