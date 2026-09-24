# Anneal — Glossary

Every concept, technique, tool and framework behind the local-first RAG app, the evaluation harness, grain-growth chunking and textreflow — what each is, how we used it, and where to learn more. The last section collects what was discussed but not used.

> Generated from the same data as the interactive version. Links go to official documentation, standards, Wikipedia or original papers; YouTube links open a search for the concept rather than one fixed video.

## Contents

- [Anneal — Local-First RAG](#anneal--local-first-rag) — 91 entries
- [Anneal — Evaluation Harness](#anneal--evaluation-harness) — 59 entries
- [Grain-Growth Chunking](#grain-growth-chunking) — 31 entries
- [Textreflow](#textreflow) — 15 entries
- [Discussed, not used](#discussed-not-used) — 30 entries

## Anneal — Local-First RAG

The application: six services that download, parse, chunk, embed and answer questions over research papers on one consumer GPU.

### Core ideas

**Retrieval-augmented generation (RAG)** · *also: Evaluation Harness*

- **What:** Answering a question by first retrieving relevant passages from a document collection, then asking a language model to answer *from those passages*. It grounds the answer in sources the model can cite.
- **How we used it:** The whole application. A question retrieves chunks from your papers, and a local model answers citing them as [1], [2]…
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) · [Lewis et al. 2020](https://arxiv.org/abs/2005.11401) · [YouTube](https://www.youtube.com/results?search_query=retrieval%20augmented%20generation%20explained)

**Local-first software**

- **What:** Software that keeps data and computation on your own machine, working offline, with the cloud optional rather than required.
- **How we used it:** Nothing in the parse or embed path touches the cloud; the model runs in Ollama on your GPU and papers never leave your disk.
- **Learn more:** [Ink & Switch essay](https://www.inkandswitch.com/local-first/) · [YouTube](https://www.youtube.com/results?search_query=local-first%20software)

**Grounding and citations**

- **What:** Constraining a model to answer only from supplied evidence and to point at which piece supports each claim.
- **How we used it:** The system prompt tells the model to use only the excerpts and cite each claim by number; the UI turns [n] into clickable chips showing the excerpt.
- **Learn more:** [Prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering) · [YouTube](https://www.youtube.com/results?search_query=LLM%20grounding%20citations%20RAG)

**Hallucination**

- **What:** A language model stating something fluent but unsupported or false.
- **How we used it:** The reason for grounding. It also explains why the model refused 'latest updates' questions — correctly — until we gave it arrival-dated papers.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)) · [YouTube](https://www.youtube.com/results?search_query=LLM%20hallucination%20explained)

### Architecture & operations

**Microservices**

- **What:** Splitting an application into small independent services that talk over the network, each owning one job.
- **How we used it:** Six services: registry (4000), embedder (4001), web UI (4002), plus parser, pruner and downloader running as background loops.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Microservices) · [YouTube](https://www.youtube.com/results?search_query=microservices%20explained)

**State machine (paper lifecycle)**

- **What:** Modelling something as a set of states with allowed transitions between them.
- **How we used it:** Every paper moves downloaded → parsed → processed → embedded (or error); the registry records the state and a timestamp per stage, so any stage can resume.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Finite-state_machine) · [YouTube](https://www.youtube.com/results?search_query=finite%20state%20machine%20explained)

**Graceful degradation**

- **What:** Keeping the system useful when a dependency is down, instead of failing entirely.
- **How we used it:** The UI still loads with the registry stopped; status endpoints report 'not reachable' rather than crashing.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Fault_tolerance) · [YouTube](https://www.youtube.com/results?search_query=graceful%20degradation%20software)

**Process management and PID files**

- **What:** Starting programs detached from the terminal and recording each one's process id in a file so it can be found and stopped later.
- **How we used it:** `ops.py` starts services detached, writes `run/pids/*.pid`, and `anneal stop` stops them by pid — which is why stop works even without the port flags.
- **Learn more:** [Process identifier](https://en.wikipedia.org/wiki/Process_identifier) · [psutil](https://psutil.readthedocs.io/) · [YouTube](https://www.youtube.com/results?search_query=linux%20process%20pid%20file%20daemon)

**Orphan processes and port ownership**

- **What:** A process left running after its parent died, often still holding a network port.
- **How we used it:** `anneal stop --all` sweeps orphans by name and by the port they hold, using psutil to find the owner.
- **Learn more:** [Orphan process](https://en.wikipedia.org/wiki/Orphan_process) · [YouTube](https://www.youtube.com/results?search_query=orphan%20process%20linux)

**Loopback binding (127.0.0.1)**

- **What:** Listening only on the local machine's address, so nothing on the network can connect.
- **How we used it:** Every service binds 127.0.0.1 — local-first in practice, and a security baseline.
- **Learn more:** [Localhost](https://en.wikipedia.org/wiki/Localhost) · [YouTube](https://www.youtube.com/results?search_query=localhost%20127.0.0.1%20explained)

**systemd user units and timers**

- **What:** Linux's service manager; a timer unit runs a service on a schedule, like cron but supervised, with catch-up for missed runs.
- **How we used it:** The daily-ingest timer (e.g. 03:00) downloads new papers and processes them; 'a missed run fires after next boot' is the Persistent setting.
- **Learn more:** [systemd](https://en.wikipedia.org/wiki/Systemd) · [systemd.timer](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html) · [YouTube](https://www.youtube.com/results?search_query=systemd%20timer%20tutorial)

**Command-line interface design**

- **What:** Building a program's commands, subcommands and flags so they are predictable and self-documenting.
- **How we used it:** The single `anneal` command: up, status, stop, fresh-start, daily-ingest, restart, with --help listing everything.
- **Learn more:** [argparse](https://docs.python.org/3/library/argparse.html) · [YouTube](https://www.youtube.com/results?search_query=python%20argparse%20tutorial)

**Environment-variable configuration**

- **What:** Configuring a program through variables in its environment instead of editing code, so every child process inherits the same settings.
- **How we used it:** ANNEAL_HOME, ANNEAL_UI_PORT/REGISTRY_PORT/EMBEDDING_PORT, VECTOR_DB_PATH, EMBED_DEVICE. `--port` flags are turned into these *before* imports, so all services agree.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Environment_variable) · [YouTube](https://www.youtube.com/results?search_query=environment%20variables%20explained)

**Separating code from data (data home)**

- **What:** Keeping everything a program writes outside its code folder, so the code can be moved, replaced or made read-only.
- **How we used it:** All papers, models, the vector store, databases, logs and pids moved to `~/.local/share/anneal`; `paths.py` is the only place that resolves it.
- **Learn more:** [XDG basedir spec](https://specifications.freedesktop.org/basedir-spec/latest/) · [YouTube](https://www.youtube.com/results?search_query=linux%20application%20data%20directory)

**Snap confinement**

- **What:** Ubuntu's snap packages run sandboxed, with some environment variables pointed at private per-snap folders.
- **How we used it:** Why the app ignores XDG_DATA_HOME: VS Code's snap terminal set it to a private folder, which would have given the app a different corpus per terminal. Also why headless Firefox couldn't write to /tmp.
- **Learn more:** [Snapcraft docs](https://snapcraft.io/docs) · [YouTube](https://www.youtube.com/results?search_query=snap%20confinement%20explained)

**nvidia-smi and VRAM**

- **What:** NVIDIA's command-line tool for GPU memory and utilisation; VRAM is the GPU's own memory.
- **How we used it:** The GPU tab, and diagnosing slow answers: the idle parser held 1.1 GB of the 4 GB card, forcing the language model 44% onto the CPU.
- **Learn more:** [nvidia-smi](https://developer.nvidia.com/system-management-interface) · [YouTube](https://www.youtube.com/results?search_query=nvidia-smi%20tutorial)

### Web backend

**FastAPI**

- **What:** A Python web framework that builds HTTP APIs from type-annotated functions, with automatic validation.
- **How we used it:** All three HTTP services — registry, embedder and web UI — are FastAPI apps.
- **Learn more:** [FastAPI](https://fastapi.tiangolo.com/) · [YouTube](https://www.youtube.com/results?search_query=fastapi%20tutorial)

**ASGI, Uvicorn and Starlette**

- **What:** ASGI is Python's interface for asynchronous web servers; Uvicorn is a server implementing it; Starlette is the toolkit FastAPI is built on.
- **How we used it:** Each service runs under Uvicorn. Starlette's StreamingResponse carries the answer stream.
- **Learn more:** [ASGI](https://asgi.readthedocs.io/) · [Uvicorn](https://www.uvicorn.org/) · [Starlette](https://www.starlette.io/) · [YouTube](https://www.youtube.com/results?search_query=ASGI%20uvicorn%20explained)

**Pydantic**

- **What:** A library that validates data against type-annotated models, turning bad input into clear errors.
- **How we used it:** Request bodies (queries, settings, prune options) are Pydantic models, so a malformed request fails loudly at the door.
- **Learn more:** [Pydantic](https://docs.pydantic.dev/) · [YouTube](https://www.youtube.com/results?search_query=pydantic%20tutorial)

**REST and JSON APIs**

- **What:** A style of HTTP API where URLs name resources and JSON carries data.
- **How we used it:** Endpoints like /v1/status, /v1/papers, /v1/prune/run, /v1/ingest/url.
- **Learn more:** [REST](https://en.wikipedia.org/wiki/REST) · [YouTube](https://www.youtube.com/results?search_query=REST%20API%20explained)

**HTTP status codes**

- **What:** Three-digit codes a server returns: 2xx success, 4xx client error, 5xx server error.
- **How we used it:** We met many: 500 (the config-collision bug), 406 (arXiv's mirror refusing PDFs), 409 (Zenodo's duplicate webhook), 429/503 (rate limits), 404 (private repo).
- **Learn more:** [MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) · [YouTube](https://www.youtube.com/results?search_query=HTTP%20status%20codes%20explained)

**Streaming responses and NDJSON**

- **What:** Sending a response piece by piece as it is produced. NDJSON is one JSON object per line, easy to parse incrementally.
- **How we used it:** The answer stream: chat id, then sources, then each generated token as a 'delta' line, then 'done'.
- **Learn more:** [NDJSON spec](https://github.com/ndjson/ndjson-spec) · [JSON streaming](https://en.wikipedia.org/wiki/JSON_streaming) · [YouTube](https://www.youtube.com/results?search_query=streaming%20HTTP%20responses%20NDJSON)

**Server-Sent Events (SSE)**

- **What:** A one-way stream from server to browser over plain HTTP, delivered as events.
- **How we used it:** The Logs tab tails each service's log live using SSE.
- **Learn more:** [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) · [YouTube](https://www.youtube.com/results?search_query=server%20sent%20events%20tutorial)

**Heartbeats and keep-alive**

- **What:** Sending small messages during silence so a connection isn't treated as dead.
- **How we used it:** The fix for 'Error in input stream': the server sends a ping line every 2 s while the model is thinking, because Firefox drops silent connections after a network change.
- **Learn more:** [Keepalive](https://en.wikipedia.org/wiki/Keepalive) · [YouTube](https://www.youtube.com/results?search_query=http%20keep%20alive%20heartbeat)

**Threads, queues and generators**

- **What:** Threads run work concurrently; a queue passes items safely between them; a generator produces values lazily and can be closed early.
- **How we used it:** The heartbeat wrapper runs the answer generator in a worker thread and forwards items through a queue, closing it cleanly (GeneratorExit) if the browser leaves.
- **Learn more:** [threading](https://docs.python.org/3/library/threading.html) · [queue](https://docs.python.org/3/library/queue.html) · [generator.close](https://docs.python.org/3/reference/expressions.html#generator.close) · [YouTube](https://www.youtube.com/results?search_query=python%20threading%20queue%20tutorial)

**Race conditions**

- **What:** A bug whose outcome depends on the timing of two things happening at once.
- **How we used it:** Deleting a chat while its answer streamed made the final save hit a missing chat (a FOREIGN KEY error); the answer now finishes and just isn't saved.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Race_condition) · [YouTube](https://www.youtube.com/results?search_query=race%20condition%20explained)

**Static file serving**

- **What:** Serving pre-built files (HTML, JS, CSS) directly rather than generating them per request.
- **How we used it:** The React app is built once into `UI/static` and served by the UI service.
- **Learn more:** [Starlette StaticFiles](https://www.starlette.io/staticfiles/) · [YouTube](https://www.youtube.com/results?search_query=serving%20static%20files%20web%20server)

### Storage

**SQLite**

- **What:** A small, serverless SQL database stored in a single file.
- **How we used it:** Two databases: the paper registry and the chat history.
- **Learn more:** [SQLite](https://www.sqlite.org/) · [YouTube](https://www.youtube.com/results?search_query=sqlite%20tutorial)

**Foreign keys and integrity errors**

- **What:** A foreign key ties a row to a row in another table; the database refuses changes that would break the link.
- **How we used it:** Chat messages reference their chat, which is how the mid-answer delete surfaced as IntegrityError.
- **Learn more:** [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html) · [YouTube](https://www.youtube.com/results?search_query=foreign%20key%20explained%20sql)

**Vector database (ChromaDB)**

- **What:** A database that stores embedding vectors and finds the nearest ones to a query vector.
- **How we used it:** ChromaDB holds 76 papers' chunks in one collection and saved chats in another, kept separate because model output isn't evidence.
- **Learn more:** [Chroma](https://docs.trychroma.com/) · [Vector database](https://en.wikipedia.org/wiki/Vector_database) · [YouTube](https://www.youtube.com/results?search_query=vector%20database%20explained)

### Retrieval & embeddings

**Embeddings** · *also: Evaluation Harness*

- **What:** Numeric vectors representing meaning, so similar texts land near each other in vector space.
- **How we used it:** Every chunk and every question is embedded; retrieval is 'find the chunks whose vectors are closest to the question's'.
- **Learn more:** [Sentence embedding](https://en.wikipedia.org/wiki/Sentence_embedding) · [YouTube](https://www.youtube.com/results?search_query=text%20embeddings%20explained)

**bge-base-en-v1.5** · *also: Evaluation Harness*

- **What:** An open English embedding model from BAAI: 768 dimensions, 512-token window.
- **How we used it:** The embedder for both the app and the evaluation. It takes a short instruction prefix on the query side only.
- **Learn more:** [Model card](https://huggingface.co/BAAI/bge-base-en-v1.5) · [C-Pack paper](https://arxiv.org/abs/2309.07597) · [YouTube](https://www.youtube.com/results?search_query=BGE%20embedding%20model)

**sentence-transformers**

- **What:** A Python library for loading and running embedding models.
- **How we used it:** Loads bge-base and embeds chunks and queries, on GPU for ingestion and CPU for daytime queries.
- **Learn more:** [SBERT](https://www.sbert.net/) · [YouTube](https://www.youtube.com/results?search_query=sentence%20transformers%20tutorial)

**Cosine distance and similarity**

- **What:** A measure of how aligned two vectors are, ignoring their length.
- **How we used it:** How Chroma ranks chunks; the 'd=0.112' shown beside each source is its distance to your question.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity) · [YouTube](https://www.youtube.com/results?search_query=cosine%20similarity%20explained)

**Top-k and approximate nearest-neighbour search**

- **What:** Returning the k most similar items; ANN indexes like HNSW find them fast without comparing against everything.
- **How we used it:** The app retrieves the top 6 chunks (4 with web search on); Chroma's index does the search.
- **Learn more:** [Nearest neighbour search](https://en.wikipedia.org/wiki/Nearest_neighbor_search) · [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) · [YouTube](https://www.youtube.com/results?search_query=HNSW%20vector%20search%20explained)

**Out-of-memory fallback**

- **What:** Catching a GPU memory error and retrying on the CPU instead of failing.
- **How we used it:** The embedder falls back to CPU if the GPU is full when it loads or searches.
- **Learn more:** [PyTorch CUDA notes](https://pytorch.org/docs/stable/notes/cuda.html) · [YouTube](https://www.youtube.com/results?search_query=CUDA%20out%20of%20memory%20fallback)

### Generation

**Large language models and tokens**

- **What:** Models that generate text one token (a word piece) at a time; the context window is how many tokens they can read at once.
- **How we used it:** The answering model reads a 6,144-token window — which is why arrival summaries cap at 12 papers.
- **Learn more:** [LLM](https://en.wikipedia.org/wiki/Large_language_model) · [Tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) · [YouTube](https://www.youtube.com/results?search_query=how%20LLM%20tokens%20work)

**Ollama and qwen3:4b-instruct**

- **What:** Ollama runs open models locally behind an HTTP API; Qwen3 4B is a small instruction-tuned model.
- **How we used it:** The default answering backend. `ollama ps` showed the model split 44% CPU / 56% GPU when the parser held VRAM.
- **Learn more:** [Ollama](https://ollama.com/) · [Qwen models](https://huggingface.co/Qwen) · [YouTube](https://www.youtube.com/results?search_query=ollama%20tutorial)

**OpenAI-compatible API**

- **What:** A widely copied HTTP interface for chat models, letting one client talk to many providers.
- **How we used it:** The optional cloud backend in Settings: any base URL, key and model that speaks this API.
- **Learn more:** [API reference](https://platform.openai.com/docs/api-reference/chat) · [YouTube](https://www.youtube.com/results?search_query=openai%20compatible%20api%20local)

**System prompts and sampling temperature**

- **What:** Instructions given to the model before the conversation; temperature controls randomness (low = more deterministic).
- **How we used it:** The system prompt sets citation rules, the date, and Mermaid label quoting; temperature is 0.2 for steady, factual answers.
- **Learn more:** [Prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering) · [YouTube](https://www.youtube.com/results?search_query=system%20prompt%20temperature%20LLM)

**Thinking-block stripping**

- **What:** Removing a model's hidden '<think>' reasoning text before showing the answer.
- **How we used it:** A filter buffers the first tokens and swallows a leaked think block.
- **Learn more:** [Qwen models](https://huggingface.co/Qwen) · [YouTube](https://www.youtube.com/results?search_query=LLM%20thinking%20tokens)

**Instruction positioning**

- **What:** Small models follow instructions placed next to the question far better than ones buried in context.
- **How we used it:** Why 'what's new' answers kept refusing until the directive travelled with the question itself, not just inside the excerpts.
- **Learn more:** [Prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering) · [YouTube](https://www.youtube.com/results?search_query=prompt%20placement%20small%20language%20models)

### Conversation features

**Conversation memory and follow-up rewriting**

- **What:** Carrying earlier turns into a new question so 'and its limitations?' has a subject.
- **How we used it:** The last few turns go into the prompt, and retrieval prepends the previous question to the search text.
- **Learn more:** [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) · [YouTube](https://www.youtube.com/results?search_query=conversational%20RAG%20follow%20up%20questions)

**Recency intent and arrival-date retrieval**

- **What:** Detecting that a question is about *when* rather than *what*, and retrieving by date instead of by meaning.
- **How we used it:** 'Latest', 'today', 'last 3 days' route to the newest embedded papers by date; the narrowest window wins; 'the 10 new updates' sets the count.
- **Learn more:** [Regular expressions](https://docs.python.org/3/library/re.html) · [YouTube](https://www.youtube.com/results?search_query=regex%20tutorial%20python)

**Web search augmentation**

- **What:** Fetching live web pages for a question and adding their text to the context.
- **How we used it:** The Web toggle: ddgs searches, trafilatura extracts each page's main text; web excerpts are cited separately from papers.
- **Learn more:** [ddgs](https://pypi.org/project/ddgs/) · [trafilatura](https://trafilatura.readthedocs.io/) · [YouTube](https://www.youtube.com/results?search_query=web%20scraping%20main%20text%20extraction)

### Parsing (recrystal)

**Document layout analysis** · *also: Evaluation Harness, Textreflow*

- **What:** Finding the regions of a page — title, text, table, figure, caption — and their positions.
- **How we used it:** Stage 1 of parsing: the detector finds 12 region types on every page.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Document_layout_analysis) · [YouTube](https://www.youtube.com/results?search_query=document%20layout%20analysis%20deep%20learning)

**Object detection and YOLO**

- **What:** Neural networks that draw labelled bounding boxes around things in an image; YOLO does it in a single pass, fast.
- **How we used it:** A fine-tuned YOLOv11 detects layout regions at 1024-pixel resolution.
- **Learn more:** [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) · [YOLO paper](https://arxiv.org/abs/1506.02640) · [YouTube](https://www.youtube.com/results?search_query=YOLO%20object%20detection%20explained)

**Fine-tuning**

- **What:** Continuing to train a pre-trained model on your own labelled data so it specialises.
- **How we used it:** The layout detector was fine-tuned on research-paper pages, including an Authors class Docling lacks.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) · [YouTube](https://www.youtube.com/results?search_query=fine%20tuning%20neural%20network%20explained)

**ONNX and ONNX Runtime**

- **What:** ONNX is a portable model format; ONNX Runtime runs it on many hardware back-ends ('execution providers').
- **How we used it:** The detector is exported to ONNX and run with onnxruntime-gpu on the CUDA execution provider — never with ultralytics at runtime, which is AGPL.
- **Learn more:** [ONNX](https://onnx.ai/) · [ONNX Runtime](https://onnxruntime.ai/) · [Execution providers](https://onnxruntime.ai/docs/execution-providers/) · [YouTube](https://www.youtube.com/results?search_query=ONNX%20runtime%20tutorial)

**Non-maximum suppression (NMS)**

- **What:** Removing overlapping duplicate boxes so each object is detected once.
- **How we used it:** Part of the ONNX detector's post-processing.
- **Learn more:** [torchvision nms](https://pytorch.org/vision/main/generated/torchvision.ops.nms.html) · [YouTube](https://www.youtube.com/results?search_query=non%20maximum%20suppression%20explained)

**PyMuPDF text extraction** · *also: Evaluation Harness*

- **What:** Reading words and their exact positions from a PDF's text layer.
- **How we used it:** Stage 1 matches PyMuPDF's words to the detector's boxes. It is also why Anneal must be AGPL.
- **Learn more:** [PyMuPDF](https://pymupdf.readthedocs.io/) · [YouTube](https://www.youtube.com/results?search_query=pymupdf%20tutorial)

**Two-stage parsing** · *also: Textreflow*

- **What:** Separating *where* things are (detection) from *how they read* (assembly).
- **How we used it:** Stage 1 = recrystal's detector; stage 2 = the assembler, now published as textreflow. The split is what let Docling's layout be run through our assembler.
- **Learn more:** [Docling](https://docling-project.github.io/docling/) · [YouTube](https://www.youtube.com/results?search_query=PDF%20parsing%20pipeline)

### Ingestion

**arXiv API and its terms of use**

- **What:** arXiv's official interface for search and metadata, with published rules on request rates.
- **How we used it:** The downloader and question-corpus builder, honouring the rules and then some: one connection, 15 s intervals, stop on 429.
- **Learn more:** [arXiv API](https://info.arxiv.org/help/api/index.html) · [Terms of use](https://info.arxiv.org/help/api/tou.html) · [YouTube](https://www.youtube.com/results?search_query=arXiv%20API%20python)

**Rate limiting and exponential backoff**

- **What:** Services cap request rates; clients wait longer after each failure (1 s, 2 s, 4 s…) instead of hammering.
- **How we used it:** Corpus building stopped rather than retry into arXiv's limiter; the downloader searches topics one at a time.
- **Learn more:** [Exponential backoff](https://en.wikipedia.org/wiki/Exponential_backoff) · [Rate limiting](https://en.wikipedia.org/wiki/Rate_limiting) · [YouTube](https://www.youtube.com/results?search_query=exponential%20backoff%20explained)

**Checkpointing and backfill**

- **What:** Remembering how far a job got so the next run continues from there; backfill fetches a window of history on first run.
- **How we used it:** Per topic, the registry returns the newest publication date it holds; with no checkpoint the downloader backfills 32 days.
- **Learn more:** [Checkpointing](https://en.wikipedia.org/wiki/Application_checkpointing) · [YouTube](https://www.youtube.com/results?search_query=incremental%20data%20ingestion%20checkpoint)

**Atomic writes**

- **What:** Writing to a temporary file, then renaming it into place in one step, so a file is either complete or absent.
- **How we used it:** The truncated-PDF fix: downloads stream into `.pdf.part` and are renamed only when complete.
- **Learn more:** [rename(2)](https://man7.org/linux/man-pages/man2/rename.2.html) · [YouTube](https://www.youtube.com/results?search_query=atomic%20file%20write%20rename)

**Streaming downloads**

- **What:** Reading a response in chunks instead of loading it whole into memory.
- **How we used it:** PDFs download in 64 KB chunks with requests' stream mode.
- **Learn more:** [requests streaming](https://requests.readthedocs.io/en/latest/user/advanced/#body-content-workflow) · [YouTube](https://www.youtube.com/results?search_query=python%20requests%20stream%20download)

**File signatures (magic bytes)**

- **What:** The first bytes that identify a file's real type, regardless of its name.
- **How we used it:** A download counts as a PDF only if it starts with `%PDF-`; an HTML error page is rejected.
- **Learn more:** [List of file signatures](https://en.wikipedia.org/wiki/List_of_file_signatures) · [YouTube](https://www.youtube.com/results?search_query=file%20magic%20number%20explained)

**Pruning and archive policy**

- **What:** Deleting intermediate files once no longer needed; archiving moves originals elsewhere instead of deleting.
- **How we used it:** Parsed/processed files go once a paper is embedded; raw PDFs are kept, archived to your external drive, or deleted, with a preview first.

### Web frontend

**React**

- **What:** A JavaScript library for building interfaces from components that re-render when their state changes.
- **How we used it:** The whole web UI.
- **Learn more:** [React](https://react.dev/) · [YouTube](https://www.youtube.com/results?search_query=react%20tutorial)

**TypeScript**

- **What:** JavaScript with static types, checked before the code runs.
- **How we used it:** All frontend code; the type checker caught the Mermaid theme bug at build time.
- **Learn more:** [TypeScript](https://www.typescriptlang.org/) · [YouTube](https://www.youtube.com/results?search_query=typescript%20tutorial)

**Vite and npm**

- **What:** npm installs JavaScript packages; Vite bundles the app into static files.
- **How we used it:** `npm run build` produces the React build the UI service serves.
- **Learn more:** [Vite](https://vite.dev/) · [npm](https://docs.npmjs.com/) · [YouTube](https://www.youtube.com/results?search_query=vite%20react%20tutorial)

**React hooks**

- **What:** Functions (useState, useEffect, useMemo, useRef) that give components state, side effects, cached values and persistent references.
- **How we used it:** Throughout the chat, welcome screen, pruning card and diagrams.
- **Learn more:** [Hooks reference](https://react.dev/reference/react/hooks) · [YouTube](https://www.youtube.com/results?search_query=react%20hooks%20explained)

**Component identity and remounting**

- **What:** React keeps a component's DOM only if its type stays the same between renders; a new function each render counts as a new type.
- **How we used it:** The zoom-dialog bug: an inline `components` object rebuilt every diagram on each re-render. `useMemo` fixed it.
- **Learn more:** [Preserving and resetting state](https://react.dev/learn/preserving-and-resetting-state) · [useMemo](https://react.dev/reference/react/useMemo) · [YouTube](https://www.youtube.com/results?search_query=react%20remount%20component%20identity)

**Markdown rendering (react-markdown, remark-gfm)**

- **What:** Turning Markdown into HTML inside React; GFM adds tables and strikethrough.
- **How we used it:** Answers render as Markdown, with [n] citations turned into chips.
- **Learn more:** [react-markdown](https://github.com/remarkjs/react-markdown) · [remark-gfm](https://github.com/remarkjs/remark-gfm) · [GFM spec](https://github.github.com/gfm/) · [YouTube](https://www.youtube.com/results?search_query=react%20markdown%20tutorial)

**Mermaid diagrams**

- **What:** A text syntax that renders flowcharts and diagrams.
- **How we used it:** Diagrams in answers render inline; broken ones get auto-quoted labels and a readable fallback; click to zoom.
- **Learn more:** [Mermaid](https://mermaid.js.org/) · [YouTube](https://www.youtube.com/results?search_query=mermaid%20diagram%20tutorial)

**Native HTML dialog**

- **What:** The browser's built-in modal element, with focus handling and Esc-to-close for free.
- **How we used it:** The diagram zoom viewer.
- **Learn more:** [MDN dialog](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dialog) · [YouTube](https://www.youtube.com/results?search_query=html%20dialog%20element)

**CSS custom properties and theming**

- **What:** Variables in CSS, redefined per theme so one stylesheet serves light and dark.
- **How we used it:** The light/dark theme and the yellow-to-blue `--hot` gradient token.
- **Learn more:** [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties) · [prefers-color-scheme](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme) · [YouTube](https://www.youtube.com/results?search_query=css%20variables%20dark%20mode)

**Scroll anchoring and auto-scroll**

- **What:** Browsers nudge scroll position as content changes; chat UIs follow new text only when the reader is at the bottom.
- **How we used it:** The fix for 'can't scroll up while answering': follow only near the bottom, instant (not smooth) scrolling, and `overflow-anchor: none`.
- **Learn more:** [overflow-anchor](https://developer.mozilla.org/en-US/docs/Web/CSS/overflow-anchor) · [YouTube](https://www.youtube.com/results?search_query=chat%20auto%20scroll%20to%20bottom%20implementation)

**Fetch streams**

- **What:** Reading an HTTP response body incrementally in the browser.
- **How we used it:** The chat reads the NDJSON answer stream line by line as it arrives.
- **Learn more:** [Streams API](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API) · [YouTube](https://www.youtube.com/results?search_query=javascript%20fetch%20readable%20stream)

**Accessibility basics**

- **What:** Making interfaces usable with keyboards and screen readers: labels, roles, reduced motion.
- **How we used it:** aria-labels on controls, visible focus, and the welcome glow respects prefers-reduced-motion.
- **Learn more:** [ARIA](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA) · [prefers-reduced-motion](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion) · [YouTube](https://www.youtube.com/results?search_query=web%20accessibility%20basics)

### Python environment

**Virtual environments** · *also: Evaluation Harness*

- **What:** An isolated Python installation per project, so dependencies don't collide.
- **How we used it:** `annealenv` for the app, and a separate throwaway `probeenv` for Docling.
- **Learn more:** [venv](https://docs.python.org/3/library/venv.html) · [YouTube](https://www.youtube.com/results?search_query=python%20virtual%20environment%20tutorial)

**Pinned requirements and dependency resolution** · *also: Evaluation Harness*

- **What:** Recording exact package versions; pip's resolver picks versions satisfying every constraint, or fails.
- **How we used it:** `app/requirements.txt` pins everything. docling-core wanting requests ≥2.34.2 against arxiv's ~=2.32 is an unsatisfiable pair — hence the separate venv.
- **Learn more:** [Requirements format](https://pip.pypa.io/en/stable/reference/requirements-file-format/) · [Dependency resolution](https://pip.pypa.io/en/stable/topics/dependency-resolution/) · [YouTube](https://www.youtube.com/results?search_query=pip%20dependency%20resolution%20explained)

**pip check**

- **What:** Verifies that installed packages' declared requirements are all satisfied.
- **How we used it:** Our standing health check after every environment change.
- **Learn more:** [pip check](https://pip.pypa.io/en/stable/cli/pip_check/) · [YouTube](https://www.youtube.com/results?search_query=pip%20check)

**The onnxruntime package collision**

- **What:** Two distributions installing into the same package folder, so whichever lands last wins.
- **How we used it:** chromadb pulls CPU onnxruntime, which silently overwrites onnxruntime-gpu; the fix is reinstalling the GPU build last with `--force-reinstall --no-deps`.
- **Learn more:** [Install ONNX Runtime](https://onnxruntime.ai/docs/install/) · [YouTube](https://www.youtube.com/results?search_query=onnxruntime%20gpu%20not%20using%20gpu)

**CUDA and PyTorch builds**

- **What:** CUDA is NVIDIA's GPU computing platform; PyTorch ships separate builds per CUDA version.
- **How we used it:** torch 2.9.1+cu128 supplies the CUDA libraries onnxruntime-gpu loads — most of the 8 GB venv.
- **Learn more:** [CUDA](https://en.wikipedia.org/wiki/CUDA) · [PyTorch](https://pytorch.org/) · [YouTube](https://www.youtube.com/results?search_query=CUDA%20explained)

**Python's import system and module caching**

- **What:** Python finds modules along sys.path and caches each by name in sys.modules; two files with the same name can shadow each other.
- **How we used it:** The 'Look first' 500 error: the pruner's config.py was cached under 'config', so the downloader got the wrong one.
- **Learn more:** [Import system](https://docs.python.org/3/reference/import.html) · [sys.modules](https://docs.python.org/3/library/sys.html#sys.modules) · [YouTube](https://www.youtube.com/results?search_query=python%20import%20system%20explained)

### Testing & verification

**pytest** · *also: Grain-Growth Chunking, Textreflow*

- **What:** Python's standard testing framework.
- **How we used it:** About 190 tests in `tests/`; run after every change.
- **Learn more:** [pytest](https://docs.pytest.org/) · [YouTube](https://www.youtube.com/results?search_query=pytest%20tutorial)

**Fixtures and monkeypatching**

- **What:** Fixtures prepare shared test setup; monkeypatch swaps a function or variable for the duration of one test.
- **How we used it:** Tests redirect the chat database to a temp file, fake the registry, and stub the model.
- **Learn more:** [Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html) · [monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html) · [YouTube](https://www.youtube.com/results?search_query=pytest%20fixtures%20monkeypatch)

**Test isolation guards**

- **What:** Tests that refuse to run if they would touch real data.
- **How we used it:** The UI tests abort if they would open your real chat database — a guard I briefly broke and then fixed.

**Smoke tests**

- **What:** A quick end-to-end check that the main path works at all.
- **How we used it:** `smoke_test.py`: parse → chunk → embed → retrieve → answer → UI → full stream, 7 stages.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Smoke_testing_(software)) · [YouTube](https://www.youtube.com/results?search_query=smoke%20testing%20explained)

**Regression tests**

- **What:** A test that pins down a fixed bug so it can't silently return.
- **How we used it:** Every bug we fixed got one: config collision, truncated downloads, mid-answer delete, heartbeat, recency.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Regression_testing) · [YouTube](https://www.youtube.com/results?search_query=regression%20testing%20explained)

**Headless browser automation**

- **What:** Driving a real browser without a window, via the WebDriver protocol, to test the UI as a user would.
- **How we used it:** geckodriver + headless Firefox reproduced the chat bugs, the zoom dialog and the scroll behaviour.
- **Learn more:** [WebDriver](https://www.w3.org/TR/webdriver2/) · [geckodriver](https://github.com/mozilla/geckodriver) · [YouTube](https://www.youtube.com/results?search_query=selenium%20webdriver%20tutorial)

**Equivalence testing** · *also: Grain-Growth Chunking, Textreflow, Evaluation Harness*

- **What:** Proving a refactor changes nothing by comparing outputs on the same inputs, ideally byte for byte.
- **How we used it:** The rename (reports byte-identical), grain-growth (same SHA-256), and textreflow (514/514 papers identical).
- **Learn more:** [SHA-2](https://en.wikipedia.org/wiki/SHA-2) · [YouTube](https://www.youtube.com/results?search_query=refactoring%20equivalence%20testing)

### Development workflow

**Git and worktrees** · *also: Evaluation Harness*

- **What:** Version control; a worktree checks out a second branch or commit beside the main one.
- **How we used it:** A worktree of the legacy commit is how round 1 compared the old parser.
- **Learn more:** [git-worktree](https://git-scm.com/docs/git-worktree) · [gitignore](https://git-scm.com/docs/gitignore) · [YouTube](https://www.youtube.com/results?search_query=git%20worktree%20tutorial)

**History rewriting**

- **What:** Changing past commits — for example to purge a leaked secret from every revision.
- **How we used it:** You rewrote history after the repo rename; I verified afterwards that the old Gemini key appears in no published commit.
- **Learn more:** [git-filter-repo](https://github.com/newren/git-filter-repo) · [YouTube](https://www.youtube.com/results?search_query=git%20remove%20secret%20from%20history)

### Licensing & security

**GNU AGPL-3.0**

- **What:** A strong copyleft licence: anyone who distributes the software, or runs a modified version as a network service, must offer its source.
- **How we used it:** Anneal's licence, inherited from PyMuPDF rather than chosen.
- **Learn more:** [Licence text](https://www.gnu.org/licenses/agpl-3.0.html) · [Wikipedia](https://en.wikipedia.org/wiki/GNU_Affero_General_Public_License) · [YouTube](https://www.youtube.com/results?search_query=AGPL%20license%20explained)

**AGPL section 13 (network use)**

- **What:** The clause requiring a network service to offer users its source code.
- **How we used it:** The small 'source' link in the web UI's top bar.
- **Learn more:** [Licence text](https://www.gnu.org/licenses/agpl-3.0.html) · [YouTube](https://www.youtube.com/results?search_query=AGPL%20network%20clause)

**SPDX licence identifiers** · *also: Grain-Growth Chunking, Textreflow*

- **What:** Short standard codes for licences, placed in file headers so tools can read them.
- **How we used it:** `AGPL-3.0-or-later` on Anneal's entry points; `Apache-2.0` on the libraries.
- **Learn more:** [SPDX list](https://spdx.org/licenses/) · [YouTube](https://www.youtube.com/results?search_query=SPDX%20license%20identifier)

**Secret handling**

- **What:** Keeping credentials out of source control and out of responses.
- **How we used it:** The keys file is git-ignored, the settings API masks secrets, and nothing hardcodes a key.
- **Learn more:** [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html) · [YouTube](https://www.youtube.com/results?search_query=secrets%20management%20best%20practices)

**Diagnosing via system logs**

- **What:** Reading journalctl and service logs to connect a symptom to its cause.
- **How we used it:** Traced the dropped streams to a crash-looping Docker container restarting every 60 s — each restart a 'network change' to Firefox.
- **Learn more:** [journalctl](https://www.freedesktop.org/software/systemd/man/latest/journalctl.html) · [YouTube](https://www.youtube.com/results?search_query=journalctl%20tutorial)

## Anneal — Evaluation Harness

The pre-registered experiment that measured parsers and chunkers over 514 papers and 400 questions, and the statistics that turned numbers into verdicts.

### Study design

**Pre-registration**

- **What:** Writing down hypotheses, metrics, tests and decision rules before seeing results, so they can't be chosen to flatter the outcome.
- **How we used it:** Round 2's plan was fixed and timestamped on 14 Sept 2026 before the run; it is why the grain-growth vs recursive splitter result is reported as 'not established'.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Preregistration_(science)) · [Nosek et al. 2018](https://doi.org/10.1073/pnas.1708274114) · [YouTube](https://www.youtube.com/results?search_query=preregistration%20explained)

**Amendment log**

- **What:** A dated, append-only record of every change to a registered plan, with what results existed at the time.
- **How we used it:** Six amendments, including the demotion of faithfulness metrics and the current→recrystal relabel.

**Frozen question set and hashing**

- **What:** Locking a dataset by recording its cryptographic hash; any change produces a different hash.
- **How we used it:** The run refuses to start if dataset.json's SHA-256 doesn't match the one in the pre-registration.
- **Learn more:** [Cryptographic hash](https://en.wikipedia.org/wiki/Cryptographic_hash_function) · [YouTube](https://www.youtube.com/results?search_query=sha256%20hash%20explained)

**Factorial design**

- **What:** Testing every combination of two or more factors.
- **How we used it:** 3 parsers × 3 chunkers = 9 cells in round 2 (12 in round 1).
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Factorial_experiment) · [YouTube](https://www.youtube.com/results?search_query=factorial%20design%20explained)

**Paired design**

- **What:** Measuring every system on exactly the same items, so differences come from the systems, not the items.
- **How we used it:** All nine cells answer the same 400 questions, which is what makes sign and Wilcoxon tests possible.
- **Learn more:** [Paired difference test](https://en.wikipedia.org/wiki/Paired_difference_test) · [YouTube](https://www.youtube.com/results?search_query=paired%20vs%20unpaired%20test)

**Statistical power and sample size**

- **What:** The chance a study detects an effect that really exists; more samples, more power.
- **How we used it:** Round 1's 100 questions were too few; it estimated 350–400. Round 2 pre-registered 400 and still could not establish the small gap.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Power_(statistics)) · [YouTube](https://www.youtube.com/results?search_query=statistical%20power%20explained)

**Replication**

- **What:** Repeating a study to see whether its result holds.
- **How we used it:** Round 2 reproduced round 1's null result at four times the question count — a replication, not a one-off.
- **Learn more:** [Replication crisis](https://en.wikipedia.org/wiki/Replication_crisis) · [YouTube](https://www.youtube.com/results?search_query=replication%20in%20science)

**Reproducibility by regeneration**

- **What:** Every number in the report is generated from raw records by a script, so anyone can rebuild it exactly.
- **How we used it:** `make_round2_report.py` regenerates Report.md byte-identically; `--round1` rebuilds the round-1 report from its own inputs.
- **Learn more:** [Reproducibility](https://en.wikipedia.org/wiki/Reproducibility) · [YouTube](https://www.youtube.com/results?search_query=reproducible%20research)

### Corpus & questions

**Corpus construction and manifests**

- **What:** Assembling a defined set of documents and recording exactly which ones.
- **How we used it:** 514 papers (7,732 pages): 95 fresh from arXiv and 419 never used in training, listed in a manifest.

**Data leakage and train–test separation**

- **What:** Testing on data the model saw during training inflates results; test data must be disjoint.
- **How we used it:** Training PDFs' arXiv ids were read off their page stamps and excluded from the evaluation corpus.
- **Learn more:** [Leakage (ML)](https://en.wikipedia.org/wiki/Leakage_(machine_learning)) · [YouTube](https://www.youtube.com/results?search_query=data%20leakage%20machine%20learning)

**LLM-generated questions with verification**

- **What:** Using a model to write questions, then checking each against the source mechanically.
- **How we used it:** gpt-oss-120b wrote 1,016 candidates; 754 passed checks against two text engines; 400 kept, one per paper.
- **Learn more:** [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) · [YouTube](https://www.youtube.com/results?search_query=synthetic%20question%20generation%20RAG%20evaluation)

**pdftotext as a neutral source**

- **What:** A plain text extractor (from Poppler) used by none of the systems under test.
- **How we used it:** Round 2 wrote questions from pdftotext output so no parser answered questions written from its own text.
- **Learn more:** [pdftotext](https://en.wikipedia.org/wiki/Pdftotext) · [Poppler](https://poppler.freedesktop.org/) · [YouTube](https://www.youtube.com/results?search_query=pdftotext%20poppler)

**Home advantage (source bias)**

- **What:** A system scores better on questions written from its own output.
- **How we used it:** Round 1 measured a 23–29 point advantage, which made parser ranking impossible there.

### Systems under test

**Parser arms**

- **What:** The parsers compared: recrystal, oss_docling (Docling layout + our assembler), oss_pymupdf4llm; round 1 also had raw_dump and legacy.
- **How we used it:** Recrystal retrieved best under every chunker and lost the fewest answers (24 of 400).
- **Learn more:** [Docling](https://docling-project.github.io/docling/) · [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) · [YouTube](https://www.youtube.com/results?search_query=document%20parsing%20comparison)

**Fixed-token chunking**

- **What:** Cutting text into windows of N tokens, usually with overlap, ignoring structure.
- **How we used it:** The weakest baseline; grain-growth beat it significantly (pooled +0.066).
- **Learn more:** [Tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) · [YouTube](https://www.youtube.com/results?search_query=fixed%20size%20chunking%20RAG)

**Recursive character splitting**

- **What:** Splitting on paragraphs, then sentences, then words, until pieces fit a size budget.
- **How we used it:** LangChain's standard splitter; grain-growth's advantage over it (+0.043) did not clear the 0.05 threshold.
- **Learn more:** [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/) · [YouTube](https://www.youtube.com/results?search_query=recursive%20character%20text%20splitter%20langchain)

**Semantic chunking**

- **What:** Splitting where the embedding similarity between adjacent sentences drops.
- **How we used it:** Measured in round 1: last at equal budget and 2.2× slower, so dropped from round 2.
- **Learn more:** [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/) · [YouTube](https://www.youtube.com/results?search_query=semantic%20chunking%20explained)

### Retrieval metrics

**Span hit rate @k**

- **What:** Whether the exact answer passage appears among the top k retrieved chunks.
- **How we used it:** span@5 reported by convention; span@4k is the primary metric.
- **Learn more:** [IR evaluation measures](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) · [YouTube](https://www.youtube.com/results?search_query=retrieval%20evaluation%20metrics)

**Budget-fair retrieval (@4,000 characters)**

- **What:** Scoring by a fixed amount of retrieved *text* rather than a fixed number of chunks.
- **How we used it:** A fixed k hands big-chunk systems more text; span_hit_near@4000ch is fair across chunkers and is the primary metric.

**Exact vs near match**

- **What:** Exact requires the verbatim span; near match accepts 90% of the span's words, tolerating text-engine differences.
- **How we used it:** Every span metric computed both ways; verdicts use near match.

**Precision@k**

- **What:** The share of the top k results that are relevant.
- **How we used it:** A secondary metric; it did not separate the chunkers.
- **Learn more:** [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) · [YouTube](https://www.youtube.com/results?search_query=precision%20at%20k%20explained)

**Mean reciprocal rank (MRR)**

- **What:** 1/rank of the first correct result, averaged; rewards putting the right answer first.
- **How we used it:** Reported, not used for verdicts.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) · [YouTube](https://www.youtube.com/results?search_query=mean%20reciprocal%20rank%20explained)

**nDCG**

- **What:** Discounted cumulative gain, normalised: graded relevance, discounted by rank.
- **How we used it:** Too flat at ~100 papers to separate cells in round 1.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) · [YouTube](https://www.youtube.com/results?search_query=nDCG%20explained)

**Metric saturation and spread**

- **What:** A metric every system aces cannot separate them; spread across cells tells you which metrics discriminate.
- **How we used it:** 'Right paper in top 5' was 1.000 for every cell; metrics with under 5 points of spread were flagged too flat.

**BEIR and TREC tradition**

- **What:** Standard benchmarks and conventions for evaluating retrieval.
- **How we used it:** Our retrieval metrics follow them.
- **Learn more:** [BEIR](https://arxiv.org/abs/2104.08663) · [YouTube](https://www.youtube.com/results?search_query=BEIR%20benchmark)

### Answer metrics

**LLM-as-a-judge**

- **What:** Using a language model to grade answers.
- **How we used it:** Gemma 3 4B judged correctness in round 2; validated against an objective numeric check (92.9% agreement).
- **Learn more:** [Zheng et al. 2023](https://arxiv.org/abs/2306.05685) · [Gemma](https://ai.google.dev/gemma) · [YouTube](https://www.youtube.com/results?search_query=LLM%20as%20a%20judge%20explained)

**Self-preference bias**

- **What:** A model rating its own output more favourably.
- **How we used it:** Round 1's answering model judged itself; round 2 separated answering from judging.

**Natural-language inference (entailment)**

- **What:** Deciding whether one text follows from, contradicts, or is neutral to another.
- **How we used it:** A 184M-parameter entailment classifier scored fact recall — the primary answer metric.
- **Learn more:** [Textual entailment](https://en.wikipedia.org/wiki/Textual_entailment) · [YouTube](https://www.youtube.com/results?search_query=natural%20language%20inference%20explained)

**Fact recall**

- **What:** The share of the reference answer's facts that the generated answer entails.
- **How we used it:** The primary answer metric; it separated no chunker.

**Faithfulness and context sufficiency (RAGAS)**

- **What:** Whether an answer is supported by its context, and whether the context held enough to answer.
- **How we used it:** Measured, but demoted before any comparison: false negatives, and identical to correctness 93% of the time.
- **Learn more:** [RAGAS paper](https://arxiv.org/abs/2309.15217) · [RAGAS docs](https://docs.ragas.io/) · [YouTube](https://www.youtube.com/results?search_query=RAGAS%20evaluation%20explained)

**Judge validation**

- **What:** Checking a judge against something objective before trusting it.
- **How we used it:** On numeric questions the check is unambiguous; the judge agreed 92.9%, fact recall 85.3%.

**Lexical overlap metrics (ROUGE-L, token F1, chrF)**

- **What:** Scores comparing answer and reference by shared words or characters.
- **How we used it:** Reported; all too flat (under 5 points of spread) to separate systems.
- **Learn more:** [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) · [F-score](https://en.wikipedia.org/wiki/F-score) · [sacreBLEU (chrF)](https://github.com/mjpost/sacrebleu) · [YouTube](https://www.youtube.com/results?search_query=ROUGE%20metric%20explained)

**Refusal rate**

- **What:** How often the model declines to answer.
- **How we used it:** 20.1% of answers declined; refusals scored as incorrect.

### Chunk shape & fidelity

**Chunk shape metrics**

- **What:** Measures of how chunks are cut: mid-sentence starts and ends, overlap, size.
- **How we used it:** Grain-growth: 1.9% of chunks start mid-sentence vs 63.9% for fixed-token under recrystal; 98% carry no overlap.

**Parse fidelity (answers lost in parsing)**

- **What:** Whether the answer text survives extraction at all — if not, no chunker can retrieve it.
- **How we used it:** recrystal lost 24 of 400 answer spans, Docling 48, PyMuPDF4LLM 61.

**Paragraph continuation metric**

- **What:** Share of paragraphs over 40 characters that begin lower case — a continuation nobody merged.
- **How we used it:** The textreflow probe: 12.5% for Docling's own model vs 2.4% through our assembler (60 papers).

### Statistics

**Null hypothesis and p-values**

- **What:** The null says there's no difference; the p-value is how surprising the data would be if that were true.
- **How we used it:** Every paired comparison produced a p-value, then went through Holm correction.
- **Learn more:** [p-value](https://en.wikipedia.org/wiki/P-value) · [Null hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis) · [YouTube](https://www.youtube.com/results?search_query=p%20value%20explained%20simply)

**Significance level (α = 0.05)**

- **What:** The threshold below which a p-value counts as significant.
- **How we used it:** Set to 0.05 before the run. Your question about reading it as 0.04 was answered from the rules: no, the verdict holds either way.
- **Learn more:** [Statistical significance](https://en.wikipedia.org/wiki/Statistical_significance) · [YouTube](https://www.youtube.com/results?search_query=significance%20level%20alpha%20explained)

**Sign test and McNemar's test**

- **What:** Tests on paired yes/no outcomes that look only at the pairs where the two systems disagree (the discordant pairs).
- **How we used it:** The binary metrics (span hit) use the exact two-sided sign test on discordant pairs — the 'coin flip' intuition we discussed.
- **Learn more:** [Sign test](https://en.wikipedia.org/wiki/Sign_test) · [McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test) · [YouTube](https://www.youtube.com/results?search_query=McNemar%20test%20explained)

**Wilcoxon signed-rank test**

- **What:** A paired test for continuous scores that uses the ranks of differences, not their raw size.
- **How we used it:** Continuous metrics like fact recall.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) · [YouTube](https://www.youtube.com/results?search_query=wilcoxon%20signed%20rank%20test%20explained)

**Multiple comparisons and family-wise error**

- **What:** Running many tests makes some 'significant' by chance; the family-wise error rate is the chance of any false positive.
- **How we used it:** Why a single p=0.022 didn't count on its own.
- **Learn more:** [Multiple comparisons](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) · [FWER](https://en.wikipedia.org/wiki/Family-wise_error_rate) · [YouTube](https://www.youtube.com/results?search_query=multiple%20comparisons%20problem%20explained)

**Holm–Bonferroni correction**

- **What:** A step-down correction: the smallest p-value faces the strictest bar, the next a looser one, and so on.
- **How we used it:** Applied within each metric's family. It's the procedure we worked through with the coin-flip analogy.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method) · [Holm 1979](https://www.jstor.org/stable/4615733) · [YouTube](https://www.youtube.com/results?search_query=Holm%20Bonferroni%20method%20explained)

**Bootstrap confidence intervals**

- **What:** Estimating uncertainty by resampling the data many times and reading off the spread of the results.
- **How we used it:** 95% percentile intervals from 10,000 resamples; drawn exactly (asymmetric) in the paper's forest plot.
- **Learn more:** [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) · [Confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) · [YouTube](https://www.youtube.com/results?search_query=bootstrap%20confidence%20interval%20explained)

**Cluster bootstrap**

- **What:** Resampling whole groups rather than individual items, so correlated items stay together.
- **How we used it:** We resampled papers, not questions.
- **Learn more:** [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) · [YouTube](https://www.youtube.com/results?search_query=cluster%20bootstrap)

**Effect size and Cohen's dz**

- **What:** How big a difference is, independent of sample size; dz is the standardised mean of paired differences.
- **How we used it:** Reported alongside p-values; a zero-variance case now returns ±∞ rather than 0.
- **Learn more:** [Effect size](https://en.wikipedia.org/wiki/Effect_size) · [YouTube](https://www.youtube.com/results?search_query=cohen's%20d%20effect%20size%20explained)

**Pooled effect and thresholds**

- **What:** The average difference across parsers, compared against a minimum size that counts as meaningful.
- **How we used it:** The deciding number: +0.043 fell below the pre-set 0.05 threshold.

**Verdict rules and fixed-sequence testing**

- **What:** Decision rules fixed in advance; metrics tested in a priority order that stops at the first failure.
- **How we used it:** Better only if significant on ≥2 of 3 parsers, no significant loss, pooled effect ≥ threshold.

**Spearman and Pearson correlation**

- **What:** Spearman measures monotonic agreement using ranks; Pearson measures linear agreement.
- **How we used it:** Shape vs outcome: Spearman −0.72 to −0.88. Judge vs fact recall: Pearson +0.66.
- **Learn more:** [Spearman](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) · [Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) · [YouTube](https://www.youtube.com/results?search_query=spearman%20vs%20pearson%20correlation)

**Correlation vs causation, confounding**

- **What:** Two things moving together doesn't mean one causes the other; a common cause can drive both.
- **How we used it:** Chunk shape correlates with retrieval, but the chunker sets both — so it can't show shape causes the gain.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation) · [Confounding](https://en.wikipedia.org/wiki/Confounding) · [YouTube](https://www.youtube.com/results?search_query=correlation%20does%20not%20imply%20causation)

### Running the experiment

**Batch-size backoff on out-of-memory**

- **What:** Retrying a GPU job with a smaller batch when it runs out of memory.
- **How we used it:** NLI scoring halves its batch on CUDA OOM (32 → 8 worked).
- **Learn more:** [PyTorch CUDA notes](https://pytorch.org/docs/stable/notes/cuda.html) · [YouTube](https://www.youtube.com/results?search_query=CUDA%20out%20of%20memory%20batch%20size)

**PyTorch expandable segments**

- **What:** An allocator setting that reduces GPU memory fragmentation.
- **How we used it:** Enabled for NLI scoring on the 4 GB card.
- **Learn more:** [PyTorch CUDA notes](https://pytorch.org/docs/stable/notes/cuda.html) · [YouTube](https://www.youtube.com/results?search_query=pytorch%20expandable%20segments)

**Resumable long runs**

- **What:** Writing progress incrementally so a crash costs minutes, not the run.
- **How we used it:** A machine failure during round 2 cost two hours, not thirty.

**Cost measurement**

- **What:** Recording time and resource use alongside quality.
- **How we used it:** Seconds per page and VRAM rise: recrystal parses 2.88× faster than Docling.

**Verified relabelling**

- **What:** Renaming a label in published data, then proving nothing else changed.
- **How we used it:** current → recrystal (Amendment 6): reports byte-identical with the label mapped back.

### Reporting & publishing

**Generated reports and SVG figures** · *also: Grain-Growth Chunking*

- **What:** Reports and charts produced by code from result files, never edited by hand.
- **How we used it:** Report.md and 16 SVG figures; `--round1` flags keep round 1 reproducible.
- **Learn more:** [SVG](https://developer.mozilla.org/en-US/docs/Web/SVG)

**Forest plot**

- **What:** A chart of effect estimates with their confidence intervals, one row per comparison.
- **How we used it:** Figure 3 in the paper: six paired comparisons against the pre-registered threshold.
- **Learn more:** [Wikipedia](https://en.wikipedia.org/wiki/Forest_plot) · [YouTube](https://www.youtube.com/results?search_query=forest%20plot%20explained)

**LaTeX, BibTeX, TikZ and pgfplots** · *also: Grain-Growth Chunking*

- **What:** LaTeX typesets documents; BibTeX formats references; TikZ/pgfplots draw figures in the source itself.
- **How we used it:** The paper, with figures drawn from the real numbers in pgfplots.
- **Learn more:** [LaTeX project](https://www.latex-project.org/) · [pgfplots](https://ctan.org/pkg/pgfplots) · [BibTeX](https://en.wikipedia.org/wiki/BibTeX) · [YouTube](https://www.youtube.com/results?search_query=latex%20tutorial%20beginners)

**Tectonic**

- **What:** A self-contained LaTeX engine that downloads only the packages a document needs.
- **How we used it:** Used to compile and verify the paper without installing TeX Live.
- **Learn more:** [Tectonic](https://tectonic-typesetting.github.io/) · [YouTube](https://www.youtube.com/results?search_query=tectonic%20latex)

**Cross-checking every number**

- **What:** Mechanically matching each figure in a write-up against its source file.
- **How we used it:** All 134 numbers in the paper checked against Report.md — which caught the symmetric-error-bar mistake.

## Grain-Growth Chunking

The chunking algorithm, and everything involved in publishing it as a citable, installable library.

### The algorithm

**Structure-aware chunking**

- **What:** Cutting documents where their own structure says, rather than where a character counter lands.
- **How we used it:** The whole library; closest prior art is Unstructured's by_title chunker.
- **Learn more:** [Unstructured chunking](https://docs.unstructured.io/open-source/core-functionality/chunking) · [YouTube](https://www.youtube.com/results?search_query=structure%20aware%20chunking%20RAG)

**Typed blocks (the input contract)**

- **What:** A document reduced to an ordered list of {type, page, text} records.
- **How we used it:** Nine types: title, section, paragraph, list, caption, table, formula, footnote, authors.

**Barriers**

- **What:** Boundaries a chunk may never cross, however much budget remains.
- **How we used it:** Section headings, and blocks that must stand alone such as tables.

**Heading-path prefix**

- **What:** Prepending 'Title › Section' to each chunk's embedded text.
- **How we used it:** A 'results' passage then embeds near a question about results in that paper.

**Next-fit bin packing**

- **What:** Filling one bin at a time and never reopening a closed one.
- **How we used it:** A chunk closes for good when the next block won't fit — one pass, reading order intact, O(n).
- **Learn more:** [Bin packing](https://en.wikipedia.org/wiki/Bin_packing_problem) · [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) · [YouTube](https://www.youtube.com/results?search_query=bin%20packing%20next%20fit)

**Sentence splitting**

- **What:** Finding sentence boundaries, which full stops alone get wrong (e.g., 'et al.').
- **How we used it:** Oversized paragraphs split at sentences with one sentence of overlap, never mid-sentence.
- **Learn more:** [Sentence boundary disambiguation](https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation) · [YouTube](https://www.youtube.com/results?search_query=sentence%20segmentation%20NLP)

**Fencing tables and formulas**

- **What:** Wrapping non-prose in labelled markers so models and readers can tell it apart.
- **How we used it:** ~~~table and ~~~formula fences; formulas stay packed with the prose that explains them.

**Formula binding**

- **What:** Keeping an equation together with its lead-in sentence.
- **How we used it:** A lead sentence without closing punctuation binds to the formula that follows.

**Annealing, recrystallisation and grain growth**

- **What:** Heat treatment: recovery, then recrystallisation forms new strain-free grains, then grain growth enlarges them until they meet boundaries.
- **How we used it:** The naming: the chunker grows chunks until a barrier; recrystal (the parser) is the stage before it.
- **Learn more:** [Annealing](https://en.wikipedia.org/wiki/Annealing_(materials_science)) · [Recrystallization](https://en.wikipedia.org/wiki/Recrystallization_(metallurgy)) · [Grain growth](https://en.wikipedia.org/wiki/Grain_growth) · [YouTube](https://www.youtube.com/results?search_query=annealing%20metallurgy%20explained)

### Library engineering

**Standard-library-only design** · *also: Textreflow*

- **What:** A library that depends on nothing but Python itself.
- **How we used it:** Only json, re, pathlib — nothing to conflict with users' environments, and no licence inherited.

**Public API and __all__** · *also: Textreflow*

- **What:** The names a library promises to keep stable; __all__ lists them, underscores mark internals.
- **How we used it:** 26 public names; internals like _FENCE_RUN stay private.
- **Learn more:** [Modules tutorial](https://docs.python.org/3/tutorial/modules.html) · [YouTube](https://www.youtube.com/results?search_query=python%20__all__%20explained)

**Input validation at the boundary** · *also: Textreflow*

- **What:** Checking inputs where they enter, failing loudly on bad data.
- **How we used it:** `validate_blocks()`: a mislabelled block wouldn't crash, it would quietly produce worse chunks.

**Adapter pattern** · *also: Textreflow*

- **What:** Small converters that translate another system's output into your input format.
- **How we used it:** Docling and Unstructured adapters, which read attributes without importing either library.
- **Learn more:** [Adapter pattern](https://en.wikipedia.org/wiki/Adapter_pattern) · [YouTube](https://www.youtube.com/results?search_query=adapter%20design%20pattern)

**pyproject.toml and src layout** · *also: Textreflow*

- **What:** The modern Python package definition file, and placing code under src/ so tests run against the installed package.
- **How we used it:** Both libraries use this layout with setuptools.
- **Learn more:** [Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) · [src vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) · [YouTube](https://www.youtube.com/results?search_query=python%20packaging%20pyproject%20toml)

**Wheels, sdists, build and twine** · *also: Textreflow*

- **What:** A wheel is a ready-to-install package; an sdist is the source archive; build makes them; twine checks and uploads.
- **How we used it:** Built with zero warnings and twine check passing, before any upload.
- **Learn more:** [Packaging guide](https://packaging.python.org/) · [build](https://build.pypa.io/) · [twine](https://twine.readthedocs.io/) · [YouTube](https://www.youtube.com/results?search_query=python%20wheel%20sdist%20explained)

**Python version floor** · *also: Textreflow*

- **What:** The oldest Python a package supports, declared in requires-python.
- **How we used it:** Verified by parsing every module with ast at Python 3.9's grammar.
- **Learn more:** [ast](https://docs.python.org/3/library/ast.html)

### Publishing

**PyPI** · *also: Textreflow*

- **What:** The Python Package Index, where pip installs from.
- **How we used it:** pip install grain-growth-chunking (and textreflow).
- **Learn more:** [PyPI](https://pypi.org/) · [YouTube](https://www.youtube.com/results?search_query=how%20to%20publish%20python%20package%20pypi)

**Trusted Publishing and OpenID Connect** · *also: Textreflow*

- **What:** GitHub proves its identity to PyPI with a short-lived signed token, so no API token is stored anywhere.
- **How we used it:** The publish workflow; configured as a 'pending publisher' before the project existed.
- **Learn more:** [Trusted publishers](https://docs.pypi.org/trusted-publishers/) · [OpenID Connect](https://en.wikipedia.org/wiki/OpenID) · [YouTube](https://www.youtube.com/results?search_query=pypi%20trusted%20publishing%20github%20actions)

**GitHub Actions and environments** · *also: Textreflow*

- **What:** CI/CD workflows triggered by repository events; environments gate deployments.
- **How we used it:** Release → build → test the wheel → publish; the 'pypi' environment; workflow_dispatch for manual runs.
- **Learn more:** [GitHub Actions](https://docs.github.com/en/actions) · [YouTube](https://www.youtube.com/results?search_query=github%20actions%20tutorial)

**Immutable versions and idempotent re-runs** · *also: Textreflow*

- **What:** PyPI never lets a version be re-uploaded; an idempotent job can run twice safely.
- **How we used it:** skip-existing: true makes re-running a release harmless — the version in pyproject decides what publishes.
- **Learn more:** [PEP 440](https://peps.python.org/pep-0440/)

**Semantic versioning** · *also: Textreflow*

- **What:** MAJOR.MINOR.PATCH, where each number signals how much changed.
- **How we used it:** Anything moving a chunk boundary takes a new major, so stored indexes stay reproducible.
- **Learn more:** [semver](https://semver.org/) · [YouTube](https://www.youtube.com/results?search_query=semantic%20versioning%20explained)

**Action runtime deprecations** · *also: Textreflow*

- **What:** Hosted CI actions depend on Node versions that GitHub retires.
- **How we used it:** The Node 20 warnings, fixed by moving to the v7 actions.
- **Learn more:** [GitHub Actions](https://docs.github.com/en/actions)

**Webhooks** · *also: Textreflow*

- **What:** An HTTP call one service makes to another when an event happens.
- **How we used it:** GitHub notifies Zenodo on release; the lingering 409 was a harmless duplicate delivery.
- **Learn more:** [Webhook](https://en.wikipedia.org/wiki/Webhook) · [YouTube](https://www.youtube.com/results?search_query=webhooks%20explained)

**Zenodo and DOIs** · *also: Textreflow*

- **What:** Zenodo (run by CERN) archives research artefacts and mints Digital Object Identifiers — permanent, citable links.
- **How we used it:** Concept DOI 10.5281/zenodo.22862100 (always latest) and version DOI …101 for 1.0.0.
- **Learn more:** [Zenodo](https://zenodo.org/) · [DOI](https://en.wikipedia.org/wiki/Digital_object_identifier) · [DataCite](https://datacite.org/) · [YouTube](https://www.youtube.com/results?search_query=zenodo%20github%20doi)

**ORCID** · *also: Textreflow*

- **What:** A permanent identifier for a researcher, separate from any identifier for their work.
- **How we used it:** 0009-0003-8880-0925, attached to both Zenodo records and in CITATION.cff and NOTICE.
- **Learn more:** [ORCID](https://orcid.org/) · [YouTube](https://www.youtube.com/results?search_query=ORCID%20explained)

**CITATION.cff** · *also: Textreflow*

- **What:** A machine-readable citation file; GitHub shows a 'Cite this repository' button and Zenodo imports it.
- **How we used it:** Its title is what Zenodo re-imports on each release — why the title-case fix went there.
- **Learn more:** [CFF](https://citation-file-format.github.io/) · [GitHub docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) · [YouTube](https://www.youtube.com/results?search_query=citation%20cff%20github)

### Licensing

**Apache-2.0** · *also: Textreflow*

- **What:** A permissive licence with an explicit patent grant.
- **How we used it:** Both libraries — possible because they link nothing AGPL.
- **Learn more:** [Licence](https://www.apache.org/licenses/LICENSE-2.0) · [Compare licences](https://choosealicense.com/licenses/) · [YouTube](https://www.youtube.com/results?search_query=apache%202.0%20license%20explained)

**Copyright vs patents**

- **What:** Copyright protects the code as written; a patent protects an invention or method however it is written.
- **How we used it:** A licence covers your code; anyone can reimplement the method from the paper.
- **Learn more:** [Patent](https://en.wikipedia.org/wiki/Patent) · [YouTube](https://www.youtube.com/results?search_query=copyright%20vs%20patent%20software)

**Patent grant and patent retaliation**

- **What:** Contributors license their relevant patents to users; anyone who sues over patents in the project loses that licence.
- **How we used it:** The deciding difference we discussed between Apache-2.0 and MIT.
- **Learn more:** [Licence text](https://www.apache.org/licenses/LICENSE-2.0) · [YouTube](https://www.youtube.com/results?search_query=apache%20license%20patent%20grant)

**NOTICE file** · *also: Textreflow*

- **What:** An attribution file Apache-2.0 requires redistributors to preserve.
- **How we used it:** Name, ORCID, DOI and a link to the evaluation — the attribution that travels with the code.

**Permissive vs copyleft**

- **What:** Permissive licences allow closed-source reuse; copyleft requires derivatives to stay open.
- **How we used it:** Libraries permissive for adoption; Anneal copyleft because PyMuPDF forces it.
- **Learn more:** [Permissive licence](https://en.wikipedia.org/wiki/Permissive_software_license) · [Copyleft](https://en.wikipedia.org/wiki/Copyleft) · [YouTube](https://www.youtube.com/results?search_query=permissive%20vs%20copyleft%20licenses)

## Textreflow

The assembler: turning a layout detector's boxes into ordered, repaired prose. Its packaging mirrors grain-growth's, so only what is specific to it is listed here.

### Assembly

**Reading-order reconstruction**

- **What:** Deciding the order a human reads a page's regions in.
- **How we used it:** Full-width bands first, then left column, then right; geometry thresholds decide what counts as full width.
- **Learn more:** [Document layout analysis](https://en.wikipedia.org/wiki/Document_layout_analysis) · [YouTube](https://www.youtube.com/results?search_query=reading%20order%20detection%20document)

**Column detection**

- **What:** Inferring page columns from region positions relative to page width.
- **How we used it:** FULL_WIDTH_FRACTION and SINGLE_COLUMN_FRACTION decide bands vs columns.

**Paragraph reconstruction**

- **What:** Rejoining a paragraph broken by a column, a page, a figure or a region boundary.
- **How we used it:** One open paragraph buffer walks regions in reading order and merges continuations.

**Continuation detection**

- **What:** Deciding whether the next text continues the last paragraph.
- **How we used it:** Text that doesn't end with terminal punctuation, followed by text starting lower case, is merged.

**De-hyphenation with a document vocabulary**

- **What:** Rejoining words split by line-wrap hyphens without breaking real hyphenated words.
- **How we used it:** Each document's own hyphenated words are collected first, so 'state-of-the-art' survives while 'recon-struction' is healed.
- **Learn more:** [Hyphen](https://en.wikipedia.org/wiki/Hyphen) · [YouTube](https://www.youtube.com/results?search_query=dehyphenation%20text%20processing)

**Page furniture removal**

- **What:** Excluding running headers, footers and text inside figures from the reading flow.
- **How we used it:** Kept as metadata in the companion JSON, dropped from the text.

**Structure tags**

- **What:** Marking structure in plain text.
- **How we used it:** # title, ## section, [AUTHORS], [CAPTION], [TABLE], [FORMULA], [FOOTNOTE].

### Input contract & adapters

**Layout JSON contract**

- **What:** Per page: width, height, and regions with a label, confidence, bounding box and lines of text.
- **How we used it:** Much heavier than grain-growth's contract — the hard part of the extraction.

**Coordinate systems (bottom-left vs top-left)**

- **What:** PDFs measure from the bottom-left corner; most image tools from the top-left.
- **How we used it:** Docling boxes are flipped to top-left in the adapter.
- **Learn more:** [PyMuPDF](https://pymupdf.readthedocs.io/) · [YouTube](https://www.youtube.com/results?search_query=pdf%20coordinate%20system%20origin)

**Docling label mapping**

- **What:** Translating Docling's region labels into the assembler's classes.
- **How we used it:** LABEL_MAP, re-exported from Anneal because the report cites it by that path.
- **Learn more:** [Docling](https://docling-project.github.io/docling/) · [YouTube](https://www.youtube.com/results?search_query=docling%20tutorial)

**Docling's .orig fallback**

- **What:** Some Docling items leave .text empty and put content in .orig.
- **How we used it:** Reading only .text silently discarded every equation; the adapter falls back to .orig.

**Tables as Markdown rows**

- **What:** Exporting a detected table's structure to Markdown lines.
- **How we used it:** Docling's TableFormer output becomes Markdown rows inside a table region.
- **Learn more:** [GFM tables](https://github.github.com/gfm/#tables-extension-) · [YouTube](https://www.youtube.com/results?search_query=markdown%20tables)

### Evidence

**The 60-paper probe**

- **What:** A measurement of how many paragraphs start mid-sentence under each pipeline.
- **How we used it:** Docling's own model 12.5% vs Docling's layout through textreflow 2.4% — five-fold, over 7,409 paragraphs.

**Pooled vs paired measurement**

- **What:** Pooling counts all items together; a paired test compares per document with an interval.
- **How we used it:** The probe is pooled, stated as a caveat; a per-paper Wilcoxon would be the stronger claim.

**Whole-corpus equivalence check**

- **What:** Running old and new code on identical inputs across the full dataset.
- **How we used it:** 514 of 514 papers byte-identical (63,783 blocks) before Anneal switched to the library.

## Discussed, not used

Ideas, tools and alternatives we talked through — considered, measured and dropped, proposed for later, or mentioned while diagnosing something.

### Packaging & distribution

**pipx**

- **What:** Installs Python command-line apps into their own isolated environments.
- **Where it came up:** Considered for making Anneal installable; would need the data home move (done) and an onnxruntime repair step.
- **Learn more:** [pipx](https://pipx.pypa.io/) · [YouTube](https://www.youtube.com/results?search_query=pipx%20tutorial)

**Docker with the NVIDIA Container Toolkit**

- **What:** Containers with GPU access.
- **Where it came up:** Would fix the CUDA/onnxruntime trap in one place, but a 10 GB+ image.
- **Learn more:** [Docker](https://docs.docker.com/) · [NVIDIA toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) · [YouTube](https://www.youtube.com/results?search_query=docker%20nvidia%20gpu)

**.deb and AppImage**

- **What:** Linux application package formats.
- **Where it came up:** Ruled out at this size.
- **Learn more:** [AppImage](https://appimage.org/) · [.deb](https://en.wikipedia.org/wiki/Deb_(file_format))

**TestPyPI**

- **What:** A practice copy of PyPI for trial uploads.
- **Where it came up:** Offered as a dry run; Trusted Publishing made it unnecessary.
- **Learn more:** [TestPyPI](https://test.pypi.org/)

**PyPI API tokens**

- **What:** A stored credential for uploading packages.
- **Where it came up:** The fallback to Trusted Publishing; not used.
- **Learn more:** [PyPI help](https://pypi.org/help/#apitoken)

**Name reservation and PEP 541**

- **What:** PyPI can reclaim abandoned or squatted names; a pending publisher holds one honestly.
- **Where it came up:** How recrystal's name is reserved without an upload.
- **Learn more:** [PEP 541](https://peps.python.org/pep-0541/)

**Hugging Face model repositories**

- **What:** Hosting model weights for download.
- **Where it came up:** The planned home for recrystal's AGPL weights, rather than a wheel.
- **Learn more:** [Hub docs](https://huggingface.co/docs/hub/models) · [YouTube](https://www.youtube.com/results?search_query=hugging%20face%20model%20hub%20upload)

### Licences & rights

**MIT licence**

- **What:** A very short permissive licence with no patent clause.
- **Where it came up:** Chosen at first for grain-growth, then replaced by Apache-2.0.
- **Learn more:** [MIT](https://opensource.org/license/mit) · [YouTube](https://www.youtube.com/results?search_query=MIT%20license%20explained)

**Public-domain dedications (CC0, Unlicense)**

- **What:** Waiving copyright entirely.
- **Where it came up:** The only way to make 'no infringement possible' true; not wanted.
- **Learn more:** [CC0](https://creativecommons.org/publicdomain/zero/1.0/) · [Unlicense](https://unlicense.org/)

**Dual licensing**

- **What:** Offering the same code under a copyleft licence and a paid commercial one.
- **Where it came up:** The only realistic route to revenue from a library; rejected in favour of adoption.
- **Learn more:** [Multi-licensing](https://en.wikipedia.org/wiki/Multi-licensing) · [YouTube](https://www.youtube.com/results?search_query=dual%20licensing%20open%20source)

**Source-available licences (BSL)**

- **What:** Free except for competing commercial use, often converting to open later.
- **Where it came up:** Mentioned as an alternative; not used.
- **Learn more:** [BSL 1.1](https://mariadb.com/bsl11/)

**CLA and DCO**

- **What:** Contributor agreements that keep the maintainer able to relicense.
- **Where it came up:** Worth adopting if you ever want dual licensing later.
- **Learn more:** [CLA](https://en.wikipedia.org/wiki/Contributor_License_Agreement) · [DCO](https://developercertificate.org/)

**GPLv2 compatibility**

- **What:** Apache-2.0 can't combine with GPLv2-only code.
- **Where it came up:** The main cost of Apache over MIT — rare in Python.
- **Learn more:** [GNU licence list](https://www.gnu.org/licenses/license-list.html)

**Trademarks**

- **What:** Rights over a name or mark, separate from copyright.
- **Where it came up:** Apache-2.0 explicitly grants none, so forks can't use your project's name.
- **Learn more:** [Trademark](https://en.wikipedia.org/wiki/Trademark)

**CC BY 4.0 for papers**

- **What:** A licence letting others reuse text with attribution.
- **Where it came up:** Suggested when submitting the paper to arXiv.
- **Learn more:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### Tools & integrations

**LangChain and LlamaIndex shims**

- **What:** Small wrappers exposing grain-growth as their TextSplitter / NodeParser.
- **Where it came up:** Offered for reach; not built.
- **Learn more:** [LlamaIndex](https://docs.llamaindex.ai/) · [LangChain splitters](https://python.langchain.com/docs/concepts/text_splitters/)

**Other layout detectors (Surya, LayoutParser, PaddleOCR)**

- **What:** Alternatives to Docling and YOLO.
- **Where it came up:** Mentioned as targets where textreflow's reassembly should help.
- **Learn more:** [Surya](https://github.com/VikParuchuri/surya) · [LayoutParser](https://layout-parser.github.io/) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

**Ultralytics at runtime**

- **What:** Running YOLO through the ultralytics package.
- **Where it came up:** Deliberately avoided: it is AGPL. Inference runs on onnxruntime instead.
- **Learn more:** [Ultralytics](https://docs.ultralytics.com/)

**SVG converters (cairosvg, rsvg)**

- **What:** Tools that turn SVG into PDF or PNG.
- **Where it came up:** Looked for to reuse report figures in the paper; figures were drawn in pgfplots instead.
- **Learn more:** [CairoSVG](https://cairosvg.org/)

**XDG_DATA_HOME**

- **What:** The standard variable for where apps store data.
- **Where it came up:** Deliberately ignored because snap terminals redirect it privately.
- **Learn more:** [XDG spec](https://specifications.freedesktop.org/basedir-spec/latest/)

### Research next steps

**arXiv submission**

- **What:** Posting the paper as a preprint.
- **Where it came up:** Drafted; cs.IR primary, cs.CL cross-list, citations to verify first.
- **Learn more:** [Submitting](https://info.arxiv.org/help/submit/index.html) · [Categories](https://arxiv.org/category_taxonomy) · [YouTube](https://www.youtube.com/results?search_query=how%20to%20submit%20paper%20to%20arXiv)

**Table- and equation-targeted questions**

- **What:** A question set whose answers sit inside tables and formulas.
- **Where it came up:** The experiment that could turn grain-growth's shape advantage into a retrieval difference — or show it doesn't.

**Per-paper Wilcoxon for textreflow**

- **What:** Keeping per-document scores so the probe gets a paired test and interval.
- **Where it came up:** Would strengthen textreflow's headline beyond pooled percentages.
- **Learn more:** [Wilcoxon](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)

**Deployment protection rules**

- **What:** Required reviewers, wait timers and branch restrictions on a GitHub environment.
- **Where it came up:** Declined for now; restricting publishing to main was offered as light hardening.
- **Learn more:** [GitHub Actions](https://docs.github.com/en/actions)

### Naming

**Printing-trade names (galley, compositor, reflow)**

- **What:** A galley proof is continuous typeset text before pagination; a compositor set type by hand.
- **Where it came up:** Candidate names for the assembler; galley and reflow were taken on PyPI, so textreflow won.
- **Learn more:** [Galley proof](https://en.wikipedia.org/wiki/Galley_proof) · [Typesetting](https://en.wikipedia.org/wiki/Typesetting)

**Metallurgy stages as names**

- **What:** Recovery → recrystallisation → grain growth.
- **Where it came up:** 'recovery' was too generic to grep; recrystal named the parser, grain growth the chunker.
- **Learn more:** [Annealing](https://en.wikipedia.org/wiki/Annealing_(materials_science))

### Things we diagnosed

**Firefox dropping silent connections**

- **What:** After a network change, Firefox closes connections that receive nothing for a few seconds.
- **Where it came up:** The root of 'Error in input stream', triggered by a crash-looping Docker container.

**Bot-challenge pages**

- **What:** Anti-bot services return a 200 'challenge' page instead of the real content.
- **Where it came up:** Why PyPI briefly looked like it held grain-growth-chunking before it existed.

**API rate limits**

- **What:** Services cap unauthenticated requests per hour.
- **Where it came up:** GitHub's API refused version lookups, so tags were read with git ls-remote instead.
- **Learn more:** [GitHub REST limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)

**Secret rotation and history purge**

- **What:** Revoking a leaked key and removing it from every commit.
- **Where it came up:** Recommended for the old Gemini key; you revoked it, and it was confirmed absent from history.
- **Learn more:** [git-filter-repo](https://github.com/newren/git-filter-repo)
