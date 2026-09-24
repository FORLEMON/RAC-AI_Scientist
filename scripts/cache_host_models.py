"""Cache fixed host assets: MiniLM, tokenizer and the ARK Conda installer."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = ("1_Pooling/config.json", "README.md", "config.json", "config_sentence_transformers.json",
         "model.safetensors", "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.txt")
MINIFORGE_SHA256 = "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05"


def download(item):
    url, path = item
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.rename(path)
    print(f"cached {path.name}", flush=True)


def main():
    assets = ROOT / "upstreams/runtime_assets"
    model = assets / "huggingface/models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = model / "snapshots" / REVISION
    jobs = [(f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}/{name}", snapshot / name) for name in FILES]
    encoding_url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    jobs.append((encoding_url, assets / "tiktoken" / hashlib.sha1(encoding_url.encode()).hexdigest()))
    miniforge = assets / "miniforge-26.7.2-0.sh"
    jobs.append(("https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-x86_64.sh", miniforge))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs))
    if hashlib.sha256(miniforge.read_bytes()).hexdigest() != MINIFORGE_SHA256:
        raise ValueError("Miniforge installer does not match its official release checksum")
    (model / "refs").mkdir(exist_ok=True)
    (model / "refs/main").write_text(REVISION, encoding="utf-8")
    inventory = {str(path.relative_to(assets)): hashlib.sha256(path.read_bytes()).hexdigest() for _, path in jobs}
    (assets / "manifest.json").write_text(json.dumps({"model_revision": REVISION, "files": inventory}, indent=2) + "\n", encoding="utf-8")
    print("host assets ready; Miniforge checksum verified", flush=True)


if __name__ == "__main__":
    main()
