#!/usr/bin/env python3
"""Export a tested local image and standalone delivery tools; never rebuild algorithms."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, text=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="roadside-perception:20260923")
    parser.add_argument("--output", required=True, type=Path, help="新的交付目录；不得已存在")
    parser.add_argument("--archive", type=Path, help="可选：复用现有 docker save 的 tar.gz，验证它对应当前镜像")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("交付目录已存在，请使用新的目录")
    info = json.loads(run("docker", "image", "inspect", args.image))[0]
    if (info["Os"], info["Architecture"]) != ("linux", "amd64"):
        parser.error("交付脚本仅支持 linux/amd64")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".roadside-release-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for name in ("load_image.sh", "check_env.sh", "check_runtime.py"):
            shutil.copy2(ROOT / "packaging" / name, staging / name)
        # Keep the standalone launcher's default in sync with the selected image.
        for name, source in (("run_container.sh", ROOT / "run_container.sh"),
                             ("check_env.sh", ROOT / "packaging/check_env.sh")):
            (staging / name).write_text(source.read_text().replace("roadside-perception:20260923", args.image))
        for script in staging.glob("*.sh"):
            script.chmod(0o755)
        shutil.copy2(ROOT / "packaging/README.md", staging / "README.md")
        (staging / "IMAGE_NAME").write_text(args.image + "\n")
        (staging / "IMAGE_ID").write_text(info["Id"] + "\n")
        env = dict(os.environ, ROAD_IMAGE=args.image)
        print("执行断网 GPU 自检……", flush=True)
        with (staging / "SELF_CHECK.log").open("w") as log:
            check = subprocess.run([str(staging / "check_env.sh")], env=env, stdout=log, stderr=subprocess.STDOUT)
        if check.returncode:
            print((staging / "SELF_CHECK.log").read_text(), flush=True)
            raise RuntimeError(f"环境自检失败: {check.returncode}")

        archive = staging / "image.tar.gz"
        if args.archive:
            print("复制现有镜像归档……", flush=True)
            shutil.copy2(args.archive.resolve(), archive)
        else:
            print("导出镜像……", flush=True)
            with archive.open("wb") as output:
                # Use the tag so docker load restores the expected image name.
                save = subprocess.Popen(["docker", "save", args.image], stdout=subprocess.PIPE)
                try:
                    subprocess.run(["gzip", "-1"], stdin=save.stdout, stdout=output, check=True)
                finally:
                    save.stdout.close()
                    code = save.wait()
                if code:
                    raise RuntimeError(f"docker save 失败: {code}")

        print("核验归档中的镜像 ID 和标签……", flush=True)
        manifest = None
        config_verified = False
        config_hash = info["Id"].split(":", 1)[1]
        with tarfile.open(archive, "r|gz") as tar:
            for entry in tar:
                name = entry.name.removeprefix("./")
                if name == "manifest.json":
                    manifest = json.load(tar.extractfile(entry))
                elif name in (f"{config_hash}.json", f"blobs/sha256/{config_hash}"):
                    config_verified = hashlib.sha256(tar.extractfile(entry).read()).hexdigest() == config_hash
        assert config_verified and manifest, "归档缺少当前镜像配置或 manifest"
        assert any(args.image in (item.get("RepoTags") or []) and
                   item["Config"].removeprefix("./") in (f"{config_hash}.json", f"blobs/sha256/{config_hash}")
                   for item in manifest), "归档标签与当前镜像 ID 不匹配"

        print("记录镜像内的依赖、源码与模型指纹……", flush=True)
        inventory_code = '''import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/app')
records={}
for p in sorted(root.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
  records[str(p.relative_to(root))]=h.hexdigest()
print(json.dumps({'python':sys.version,'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True).splitlines(),'app_sha256':records},ensure_ascii=False,indent=2))
'''
        inventory = run("docker", "run", "--rm", "--network", "none", "--entrypoint", "python", info["Id"], "-c", inventory_code)
        (staging / "IMAGE_CONTENTS.json").write_text(inventory)
        (staging / "IMAGE_INSPECT.json").write_text(json.dumps(info, indent=2))
        build_info = {
            "release_created_utc": datetime.now(timezone.utc).isoformat(),
            "image": args.image, "image_id": info["Id"], "image_created": info["Created"],
            "platform": f'{info["Os"]}/{info["Architecture"]}', "image_size_bytes": info["Size"],
            "packaging_source_commit": run("git", "-C", str(ROOT), "rev-parse", "HEAD").strip(),
            "packaging_source_status": run("git", "-C", str(ROOT), "status", "--short"),
            "note": "Existing image reused. Git information describes packaging workspace, not proven image source commit. Image contents are fingerprinted separately.",
            "self_check": "SELF_CHECK.log; full video accuracy/long-duration acceptance is separate",
        }
        (staging / "BUILD_INFO.txt").write_text(json.dumps(build_info, ensure_ascii=False, indent=2) + "\n")
        print("生成 SHA256SUMS……", flush=True)
        sums = [f"{sha256(p)}  {p.name}\n" for p in sorted(staging.iterdir()) if p.is_file()]
        (staging / "SHA256SUMS").write_text("".join(sums))
        # Rename only after all validation succeeds; retain no partial release.
        staging.rename(destination)
    print(f"交付目录已生成: {destination}", flush=True)


if __name__ == "__main__":
    main()
