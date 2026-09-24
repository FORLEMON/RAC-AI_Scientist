"""Install the three active hosts from pinned GitHub source archives.

Useful when Git smart-HTTP is unavailable. Existing sources are verified,
never replaced. No framework dependencies are installed by this command.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rac_ai_scientist.provenance import tree_hash, IGNORED_PARTS, IGNORED_SUFFIXES, _is_generated_package_metadata


def extract_archive(archive, target):
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 1:
                raise ValueError("unsafe source archive path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("source archive symlink requires a native Git checkout")
        target.mkdir(parents=True, exist_ok=False)
        for entry in bundle.infolist():
            parts = PurePosixPath(entry.filename).parts[1:]
            if not parts or entry.is_dir():
                continue
            destination = target.joinpath(*parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError("source archive path escapes destination")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.read(entry))
            mode = entry.external_attr >> 16
            if mode:
                destination.chmod(mode & 0o777)


def install_source(name, spec, cache):
    target = ROOT / "upstreams" / name
    if not target.exists():
        archive = cache / f"{name}-source.zip"
        if not archive.exists():
            repository = spec["url"].removeprefix("https://github.com/").removesuffix(".git")
            urllib.request.urlretrieve(f"https://codeload.github.com/{repository}/zip/{spec['revision']}", archive)
        extract_archive(archive, target)
    actual, count, size = tree_hash(target)
    if actual != spec["tree_sha256"]:
        raise ValueError(f"{name} source hash mismatch: {actual}")
    print(f"[OK] {name} revision={spec['revision']} files={count} bytes={size} sha256={actual}", flush=True)
    if name == "ark":
        # The upstream .dockerignore excludes venue_templates and skills. A
        # verified source tar restores the full tree inside the research image
        # without changing the pristine checkout or its .dockerignore.
        archives = ROOT / "upstreams/host_archives"
        archives.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archives / "ark-source.tar", "w") as bundle:
            for path in sorted(target.rglob("*")):
                relative = path.relative_to(target)
                if (not path.is_file() or IGNORED_PARTS.intersection(relative.parts)
                        or _is_generated_package_metadata(relative) or path.suffix in IGNORED_SUFFIXES):
                    continue
                info = bundle.gettarinfo(str(path), arcname=relative.as_posix())
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    bundle.addfile(info, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("ark", "agent_laboratory", "evo_scientist"), action="append")
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    cache = ROOT / "upstreams/host_api"
    cache.mkdir(parents=True, exist_ok=True)
    for name in args.only or ("ark", "agent_laboratory", "evo_scientist"):
        spec = lock["upstreams"][name]
        install_source(name, spec, cache)
        for submodule, dependency in spec.get("submodules", {}).items():
            install_source(submodule, dependency, cache)


if __name__ == "__main__":
    main()
