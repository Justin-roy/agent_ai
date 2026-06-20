# Build Your Own Local AI Agent — From Scratch

A complete, hands-on guide to running an open-source LLM locally and turning it
into a tool-using **AI agent** — including Docker. Written specifically for an
**Acer Nitro 5 (Intel i5, 16 GB RAM, GTX 1650 4 GB)**.

---

## 0. Reality check (read this first — it shapes everything)

Your **GTX 1650 has 4 GB of VRAM**. That single number decides which models you
can run. Most "run an LLM locally" tutorials quietly assume 8–24 GB, so ignore
the ones that tell you to run a 13B model.

What that means in practice:

| Model size (4-bit) | Fits in 4 GB VRAM? | Verdict for you |
|---|---|---|
| 1B (e.g. `llama3.2:1b`) | Yes, easily | Very fast, weak reasoning |
| 1.5B–3B (`qwen2.5:3b`, `llama3.2:3b`) | Yes | **Sweet spot** — fast enough, good enough at tools |
| 7B–8B (`llama3.1:8b`) | No (spills to RAM) | Works but slow (CPU offload via your 16 GB RAM) |
| 13B+ | No | Don't bother on this laptop |

**Recommended model: `qwen2.5:3b`.** Among small models it's one of the best at
*tool/function calling*, which is exactly what an agent needs. Keep
`llama3.2:1b` around for when you want speed.

Your 16 GB of system RAM is your safety net: when a model doesn't fully fit in
VRAM, Ollama automatically runs the leftover layers on the CPU using that RAM.
Slower, but it works.

---

## 1. The mental model — what an "AI agent" actually is

Strip away the hype and an agent is just **four parts**:

1. **A model runtime** — a program that loads the LLM and answers requests.
   We'll use **Ollama** (handles GPU offload, quantization, and an API for you).
2. **The model** — the "brain". `qwen2.5:3b`.
3. **Tools** — ordinary functions the model is allowed to call (a calculator,
   a web search, reading a file...). These are the agent's "hands".
4. **The agent loop** — the orchestrator. It sends the user's message to the
   model; if the model says "call tool X with these args", the loop runs the
   tool, feeds the result back, and repeats until the model gives a final answer.

> **Agent = LLM + tools + a loop that lets it act.** That's the whole secret.
> The popular frameworks (LangChain, LlamaIndex, smolagents) are just nicer
> wrappers around that loop. We'll build the loop ourselves first so you
> understand it, then you can adopt a framework later if you want.

The classic version of this loop is called **ReAct** (Reason + Act): the model
*reasons* about what to do, *acts* by calling a tool, observes the result, and
repeats. Modern models do this through structured "tool calls" instead of
free text, which is what our code uses.

---

## 2. The roadmap

- **Phase 1** — Install Ollama, run a model, confirm the GPU is working. *(fast win)*
- **Phase 2** — Build a minimal agent in Python (no framework).
- **Phase 3** — Containerize everything with Docker + GPU.
- **Phase 4** — Where to go next.

Do Phase 1 and 2 **on bare metal first**. Get a working agent before adding the
complexity of Docker. Docker is for reproducibility and deployment, not for
your first "hello world".

---

## 3. Phase 1 — Run an open-source LLM locally

### 3.1 First, sort out your GPU driver

You need a recent NVIDIA driver. Open a terminal and run:

```bash
nvidia-smi
```

You should see a table listing your GTX 1650 and a driver/CUDA version. If the
command isn't found, install the latest **NVIDIA Game Ready / Studio driver**
for the GTX 1650 from nvidia.com (Windows) or your distro's driver tool (Linux).
You do **not** need to install the full CUDA Toolkit separately — Ollama ships
the CUDA libraries it needs.

### 3.2 Install Ollama

**Windows:** download the installer from <https://ollama.com/download> and run
it. It installs a background service.

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Working in WSL2 (Windows Subsystem for Linux)?** This matters later for
Docker. If you plan to use Docker with GPU on Windows, do your Linux-style work
inside WSL2 (Ubuntu). GPU access flows: *Windows NVIDIA driver → WSL2 → Docker*.
You install the driver on **Windows only**, never inside WSL2.

### 3.3 Pull and run your first model

```bash
ollama pull qwen2.5:3b      # downloads ~2 GB, one time
ollama run qwen2.5:3b       # opens an interactive chat
```

Type a question, press Enter, and you're talking to a local LLM with **no
internet, no API key, no cost**. Type `/bye` to exit.

### 3.4 Confirm the GPU is actually being used

In one terminal start the model, and in another run:

```bash
ollama ps
```

The `PROCESSOR` column tells you the split, e.g. `100% GPU` or `70%/30%
CPU/GPU`. While generating, `nvidia-smi` should show VRAM in use and the GPU
working. If you see `100% CPU`, the model is too big for VRAM — switch to a
smaller one or reduce context (see Troubleshooting).

✅ **Checkpoint:** you can chat with a local model and the GPU is engaged.

---

## 4. Phase 2 — Build the agent (Python, no framework)

Ollama exposes an HTTP API on `http://localhost:11434`. Our agent will:
send the conversation → if the model requests a tool, run it → feed the result
back → repeat until done.

### 4.1 Set up a project

```bash
mkdir local-agent && cd local-agent
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install requests
```

> On your laptop, prefer **Python 3.10+**. The `venv` keeps this project's
> packages isolated from the rest of your system.

### 4.2 The agent code

Use the `agent.py` included alongside this guide. It's fully commented, but
here's the heart of it — the loop that makes it an agent:

```python
def run_agent(user_input, max_steps=8):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
        {"role": "user", "content": user_input},
    ]
    for _ in range(max_steps):
        msg = chat(messages)            # one call to the local model
        messages.append(msg)
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")   # model gave a final answer
        for call in tool_calls:             # model wants to use a tool
            name = call["function"]["name"]
            args = call["function"].get("arguments", {}) or {}
            result = TOOLS[name](**args)     # run the real Python function
            messages.append({"role": "tool", "name": name, "content": str(result)})
```

The tools themselves are just normal functions (a safe calculator and a clock
in the starter), each described to the model with a small JSON schema so it
knows the name, purpose, and arguments.

### 4.3 Run it

Make sure Ollama is running, then:

```bash
python agent.py
```

Try these — the first needs no tool, the others should trigger one:

```
You: Hello, who are you?
You: What is 18.5% of 240?
You: What time is it right now?
```

When a tool fires you'll see a `[tool] ...` line showing exactly what the model
called and what came back. That transparency *is* the learning.

### 4.4 Add your own tool (the important exercise)

To extend the agent, you do two things: write the function, and describe it.

```python
def reverse_text(text: str) -> str:
    return text[::-1]

# add to TOOLS:
TOOLS["reverse_text"] = reverse_text

# add to TOOL_SCHEMAS:
{
  "type": "function",
  "function": {
    "name": "reverse_text",
    "description": "Reverse a string of text.",
    "parameters": {
      "type": "object",
      "properties": {"text": {"type": "string", "description": "Text to reverse."}},
      "required": ["text"],
    },
  },
}
```

That pattern — *function + schema* — is how every tool gets added, from a
calculator to a web search to "send an email". The model can only do what you
give it a tool for.

✅ **Checkpoint:** you have a working agent that reasons and calls tools.

---

## 5. Phase 3 — Containerize with Docker (+ GPU)

Now we make it reproducible. The setup: **two containers** — Ollama (serving the
model, using your GPU) and your agent (the Python app) — wired together so the
agent reaches Ollama by name.

### 5.1 Install Docker

- **Windows:** install **Docker Desktop** and enable the **WSL2 backend**
  (Settings → General → "Use WSL2 based engine"). This is what lets containers
  see your GPU on Windows.
- **Linux:** install Docker Engine from <https://docs.docker.com/engine/install/>.

Verify: `docker run hello-world`.

### 5.2 Let Docker see the GPU (the step everyone trips on)

**Linux** needs the **NVIDIA Container Toolkit**:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows (Docker Desktop + WSL2):** no toolkit install needed — GPU support
comes through WSL2 automatically, *provided* your Windows NVIDIA driver is
recent and the WSL2 backend is on.

**Test GPU-in-Docker** (both platforms):

```bash
docker run --rm --gpus all ollama/ollama nvidia-smi
```

If you see your GTX 1650 listed, you're done with the hard part.

### 5.3 Run Ollama in a container

```bash
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull qwen2.5:3b
```

The `-v ollama:/root/.ollama` volume means downloaded models survive container
restarts — you only pull once.

### 5.4 Tie it together with docker-compose

Use the included `Dockerfile`, `docker-compose.yml`, `requirements.txt`, and
`agent.py`. From the project folder:

```bash
# 1. Start Ollama (in the background)
docker compose up -d ollama

# 2. Pull the model into the Ollama container (one time)
docker compose exec ollama ollama pull qwen2.5:3b

# 3. Run the agent interactively
docker compose run --rm agent
```

Inside compose, the agent reaches Ollama at `http://ollama:11434` (the service
name), set via the `OLLAMA_URL` environment variable. To swap models, just
change the `MODEL` value in `docker-compose.yml`.

> Tip: use `docker compose run --rm agent` (not `up`) because the agent is an
> *interactive* chat — `run` attaches your terminal to it.

✅ **Checkpoint:** the whole stack runs in containers, GPU-accelerated, and
you can rebuild it anywhere with one command.

---

## 6. Phase 4 — Where to go next

Now that the foundation works, here's the natural progression:

- **More useful tools.** A web search tool (e.g. via a search API), a file
  reader, a shell command runner, a calendar lookup. Each is the same
  *function + schema* pattern.
- **Memory.** Right now each `run_agent` call starts fresh. Keep the `messages`
  list across turns to give the agent conversational memory.
- **RAG (retrieval).** Let the agent answer questions about *your* documents:
  embed your files into a vector store (e.g. ChromaDB), retrieve relevant chunks,
  and feed them into the prompt. This is how you get a "chat with my PDFs" agent
  — and it runs fine on your hardware since embeddings are cheap.
- **Streaming output.** Set `"stream": true` in the API call and read tokens as
  they arrive, so responses appear word-by-word.
- **Adopt a framework (optional).** Once you understand the loop, tools like
  **smolagents** (lightweight, great for learning), **LangChain**, or
  **LlamaIndex** save boilerplate. You'll appreciate them *more* having built
  the loop by hand.
- **Try bigger models occasionally.** `llama3.1:8b` will run on your laptop via
  CPU offload — slow, but useful to feel the quality difference. Stick with 3B
  for anything interactive.

---

## 7. Troubleshooting

**"out of memory" / model loads as `100% CPU`**
VRAM is full. Use a smaller model (`llama3.2:1b`, `qwen2.5:1.5b`) or shrink the
context window. Create a custom variant with a smaller context:

```bash
printf 'FROM qwen2.5:3b\nPARAMETER num_ctx 2048\n' > Modelfile
ollama create qwen2.5:3b-small -f Modelfile
```

**Generation is very slow**
You're running on CPU. Confirm with `ollama ps`. Switch to a 1B–3B model and
make sure the GPU driver is current. 4 GB VRAM means small models are the right
call — that's not a failure, it's the hardware.

**`docker: could not select device driver ... gpu`**
Docker can't see the GPU. Linux: re-run the NVIDIA Container Toolkit steps and
restart Docker. Windows: update the NVIDIA driver and confirm the WSL2 backend
is enabled in Docker Desktop. Then re-test with the `nvidia-smi` container.

**Agent can't reach Ollama**
Bare metal: is Ollama running? (`ollama ps`). In Docker: the agent must use
`http://ollama:11434`, not `localhost` — inside a container `localhost` means
the container itself.

**The model ignores tools or calls them wrong**
Small models aren't perfect at tool use. Prefer `qwen2.5:3b`, keep tool
descriptions short and clear, and don't give it too many tools at once.

---

## Quick command reference

```bash
# Models
ollama pull qwen2.5:3b          # download
ollama run qwen2.5:3b           # chat directly
ollama list                     # what's installed
ollama ps                       # what's running + CPU/GPU split

# Agent (bare metal)
python agent.py

# Docker stack
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:3b
docker compose run --rm agent
docker compose down             # stop everything

# Python UI (Run on other terminal)
pip install streamlit
pip install -r requirements.txt
streamlit run app.py
```

---

*Built for a GTX 1650 (4 GB). The principles scale: the same code runs a 70B
model on a bigger GPU — you'd just change one line, `MODEL=...`.*
