import asyncio
import aiohttp
import aiofiles
import hashlib
import json
import yaml
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


try:
    CONFIG = load_yaml("configs/data.yaml")["scraper"]
except Exception as e:
    print(f"Не удалось загрузить конфиг scraper из data.yaml: {e}")
    exit(1)

IMAGE_PATH = Path(CONFIG["output_dir"]) / "images"
METADATA_FILE = Path(CONFIG["output_dir"]) / "dataset.jsonl"
SEMAPHORE = asyncio.Semaphore(CONFIG["concurrency"])


async def fetch_and_save(session, pbar):
    api_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"

    async with SEMAPHORE:
        try:
            async with session.get(api_url, timeout=10) as resp:
                if resp.status != 200:
                    return

                data = await resp.json()

                if "originalimage" not in data or "extract" not in data:
                    return
                if len(data["extract"]) < CONFIG["min_text_len"]:
                    return

                img_url = data["originalimage"]["source"]
                if img_url.lower().endswith(".svg"):
                    return

                async with session.get(img_url, timeout=15) as img_resp:
                    if img_resp.status != 200:
                        return
                    img_data = await img_resp.read()

                try:
                    with Image.open(BytesIO(img_data)) as img:
                        img.verify()
                except:
                    return

                file_hash = hashlib.md5(data["title"].encode()).hexdigest()
                ext = img_url.split(".")[-1].split("?")[0][:4]
                filename = f"{file_hash}.{ext}"

                async with aiofiles.open(IMAGE_PATH / filename, "wb") as f:
                    await f.write(img_data)

                res = {
                    "title": data["title"],
                    "text": data["extract"],
                    "image_file": filename,
                }

                async with aiofiles.open(
                    METADATA_FILE, "a", encoding="utf-8"
                ) as f_meta:
                    await f_meta.write(json.dumps(res, ensure_ascii=False) + "\n")

                pbar.update(1)

        except Exception:
            pass


async def worker(session, pbar):
    while True:
        current_count = 0
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r") as f:
                current_count = sum(1 for _ in f)

        if current_count >= CONFIG["target_count"]:
            break

        await fetch_and_save(session, pbar)


async def main():
    IMAGE_PATH.mkdir(parents=True, exist_ok=True)

    start_count = 0
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            start_count = sum(1 for _ in f)

    pbar = tqdm(total=CONFIG["target_count"], initial=start_count, desc="Scraping Wiki")

    headers = {"User-Agent": CONFIG["user_agent"]}
    connector = aiohttp.TCPConnector(limit=50, force_close=True)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        workers = [worker(session, pbar) for _ in range(10)]
        await asyncio.gather(*workers)

    pbar.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(":(")
