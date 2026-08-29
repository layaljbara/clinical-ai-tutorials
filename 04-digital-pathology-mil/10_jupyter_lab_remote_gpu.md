# 10 — Jupyter Lab on a remote GPU machine

Run notebooks **inside the same conda env** as OpenSlide, PyTorch, and your encoder.

Do **not** mix system `apt install jupyter` with a conda PyTorch stack unless you know what you are doing.

## Install (once)

```bash
ssh user@YOUR_GPU_HOST
conda activate wsi-mil   # your env name
conda install -c conda-forge jupyterlab
# or: pip install jupyterlab
which jupyter
jupyter --version
```

## Start Lab (no browser on the server)

```bash
cd /path/to/your_project
conda activate wsi-mil
jupyter lab --no-browser --port=8888
```

Copy the URL with token from the terminal.

## Open from your laptop (SSH tunnel)

In a **second** local terminal:

```bash
ssh -N -L 8888:127.0.0.1:8888 user@YOUR_GPU_HOST
```

Then open `http://127.0.0.1:8888/lab?token=...` in your browser.

## Tips

- Put data paths in environment variables or a local `config.yaml` (not in git).  
- Keep large WSIs on the server filesystem, not in the notebook repo.  
- Restart kernel after `conda install` changes.

Next: [11_embedding_visualization.md](11_embedding_visualization.md)
