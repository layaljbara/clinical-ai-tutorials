# Clinical AI tutorials

General teaching materials on **Python**, **APIs & literature mining**, **transformers / LLMs**, and **digital pathology / MIL** — focused on **libraries and concepts**, not on any one research project.

**Author:** [Layal Jbara](https://layaljbara.github.io)  
**Site:** [layaljbara.github.io/teaching](https://layaljbara.github.io/teaching/)

No patient data, whole-slide images, or private project code.

## Modules

| Folder | What you learn |
|--------|----------------|
| [`01-python-fundamentals/`](01-python-fundamentals/) | OOP, files/paths, errors, generators, regex |
| [`02-apis-and-literature/`](02-apis-and-literature/) | HTTP, NCBI/Entrez, Hugging Face datasets, PMC XML |
| [`03-transformers-and-llms/`](03-transformers-and-llms/) | Transformers, MoE/reasoning intro, LLM-from-scratch lectures |
| [`04-digital-pathology-mil/`](04-digital-pathology-mil/) | OpenSlide, tiling, tile encoders, attention-MIL (generic) |

## Design principles

- **Library-first** — teaches OpenSlide, PyTorch, `requests`, `timm`, etc.  
- **Project-agnostic** — example queries and labels are illustrative only  
- **Bring your own data** — slides, notes, and checkpoints stay outside this repo  

## Prerequisites

- Python 3.10+  
- Module 02: NCBI email + API key via environment variables  
- Module 04: OpenSlide system library + optional GPU for embedding demos  

## License

MIT — see [`LICENSE`](LICENSE).
