import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import logging, warnings

INDEX_REFRESH_TTL = 3600 * 24  # раз в день

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Accessing `__path__`")
warnings.filterwarnings("ignore", message="resource_tracker")

import json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import faiss
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor

BACKENDS_CONFIG_PATH = "backends.yaml"

T = {
    "en": {
        "page_title": "WikiLens — Shazam for Wikipedia",
        "hero_sub": "Point your camera at anything. Get the Wikipedia article.",
        "upload_label": "**Drop a photo**",
        "upload_caption": "JPG, PNG, WEBP · drag anywhere on the page",
        "model_label": "**Model**",
        "index_label": "**Search index**",
        "sidebar_title": "⚙️ Settings",
        "filter_label": "Result mode",
        "filter_top": "Top N articles",
        "filter_thresh": "Min confidence",
        "topk_label": "Number of results",
        "thresh_label": "Minimum confidence, %",
        "thresh_help": "Hide results below this threshold.",
        "models_expander": "📖 About models",
        "index_expander": "🔀 About index modes",
        "results_title": "### Results",
        "read_more": "Read on Wikipedia →",
        "match": "match",
        "no_image": "no image",
        "no_results": "No results above {pct}% confidence. Try lowering the threshold.",
        "open_error": "Could not open file: {e}",
        "model_error": "Could not load model: `{e}`",
        "search_error": "Search error: {e}",
        "config_error": "Config `{path}` not found.",
        "searching": "Searching…",
        "placeholder": "← Drop a photo to search",
        "downloading": "⬇️ Downloading {name}…",
        "warming": "⏳ Loading models {i}/{n}…",
        "hero_desc": "Snap a photo of any object — an animal, building, plant, or artifact — and instantly get the Wikipedia article about it. Like Shazam, but for the visual world.",
        "model_disclaimer_title": "💡 About model selection",
        "model_disclaimer": (
            "Model switching is primarily useful for comparing training approaches, "
            "or finding a different perspective if the default model doesn't give good results. "
            "For everyday use we recommend keeping the default — it performs best overall."
        ),
        "index_disclaimer_title": "🔀 About index selection",
        "index_disclaimer": (
            "Index switching is useful for comparison or if you want a different angle on the results. "
            "**RRF is the default and works best** — for most photos just leave it as is. "
            "The other modes exist for experimentation or edge cases."
        ),
        "contacts_title": "Made by",
        # fusion picker labels
        "fusion_rrf": "🔀 RRF — both indices",
        "fusion_combined": "⚖️ Combined (score merge)",
        "fusion_image": "🖼 Image index only",
        "fusion_text": "📝 Text index only",
        "fusion_no_text_index": "⚠️ Text index not configured for this model — falling back to image index only.",
        # sidebar index descriptions
        "idx_desc_rrf": (
            "**🔀 RRF — both indices**\n\n"
            "Combines the image index and the text index using Reciprocal Rank Fusion. "
            "Results are ranked by their combined position in both lists. Best overall."
        ),
        "idx_desc_combined": (
            "**⚖️ Combined (score merge)**\n\n"
            "Same two indices, but merged by raw cosine similarity score. "
            "The highest-scoring result from either index wins."
        ),
        "idx_desc_image": (
            "**🖼 Image index only**\n\n"
            "Searches purely in the visual embedding space. "
            "Good when the photo has a distinctive appearance — a landmark, a specific animal species."
        ),
        "idx_desc_text": (
            "**📝 Text index only**\n\n"
            "Searches in the semantic text embedding space. "
            "Useful when the photo depicts a concept or scene rather than a unique object."
        ),
    },
    "ru": {
        "page_title": "WikiLens — Шазам для Википедии",
        "hero_sub": "Сфотографируйте что угодно — получите статью из Википедии.",
        "upload_label": "**Бросьте фото**",
        "upload_caption": "JPG, PNG, WEBP · тащите в любое место страницы",
        "model_label": "**Модель**",
        "index_label": "**Индекс поиска**",
        "sidebar_title": "⚙️ Настройки",
        "filter_label": "Режим показа",
        "filter_top": "Топ N статей",
        "filter_thresh": "Мин. уверенность",
        "topk_label": "Количество результатов",
        "thresh_label": "Минимальное совпадение, %",
        "thresh_help": "Скрыть результаты ниже порога.",
        "models_expander": "📖 О моделях",
        "index_expander": "🔀 О режимах индекса",
        "results_title": "### Вот что нашлось",
        "read_more": "Читать на Википедии →",
        "match": "совпадение",
        "no_image": "нет изображения",
        "no_results": "Нет результатов выше {pct}%. Попробуйте снизить порог.",
        "open_error": "Не удалось открыть файл: {e}",
        "model_error": "Не удалось загрузить модель: `{e}`",
        "search_error": "Ошибка поиска: {e}",
        "config_error": "Файл `{path}` не найден.",
        "searching": "Ищем статьи…",
        "placeholder": "← Бросьте фото для поиска",
        "downloading": "⬇️ Скачиваем {name}…",
        "warming": "⏳ Загружаем модели {i}/{n}…",
        "hero_desc": "Сфотографируйте любой предмет — животное, здание, растение или артефакт — и мгновенно получите статью из Википедии. Как Shazam, только для визуального мира.",
        "model_disclaimer_title": "💡 О выборе модели",
        "model_disclaimer": (
            "Выбор модели нужен скорее для сравнения методов обучения или другого взгляда, "
            "если основная модель не выдаёт хороших результатов. "
            "Для обычного использования рекомендуем оставить модель по умолчанию — она строго лучше остальных."
        ),
        "index_disclaimer_title": "🔀 О выборе индекса",
        "index_disclaimer": (
            "Выбор индекса нужен для сравнения или другого взгляда на результаты. "
            "**RRF по умолчанию работает лучше всего** — для большинства фото лучше оставить как есть. "
            "Остальные режимы — для экспериментов или конкретных случаев."
        ),
        "contacts_title": "Сделал",
        # fusion picker labels
        "fusion_rrf": "🔀 RRF — оба индекса",
        "fusion_combined": "⚖️ Объединённый (по score)",
        "fusion_image": "🖼 Только индекс фото",
        "fusion_text": "📝 Только индекс текста",
        "fusion_no_text_index": "⚠️ Текстовый индекс не настроен для этой модели — используется только индекс фото.",
        # sidebar index descriptions
        "idx_desc_rrf": (
            "**🔀 RRF — оба индекса**\n\n"
            "Объединяет индекс фото и индекс текстов через Reciprocal Rank Fusion — "
            "итоговый ранг строится по позиции в обоих списках одновременно. Лучший вариант в целом."
        ),
        "idx_desc_combined": (
            "**⚖️ Объединённый (по score)**\n\n"
            "Те же два индекса, но слияние по значению косинусного сходства — "
            "побеждает результат с наивысшим score из любого из индексов."
        ),
        "idx_desc_image": (
            "**🖼 Только индекс фото**\n\n"
            "Поиск исключительно в визуальном пространстве эмбеддингов. "
            "Хорошо работает, когда на фото узнаваемый объект — здание, конкретный вид животного."
        ),
        "idx_desc_text": (
            "**📝 Только индекс текста**\n\n"
            "Поиск в семантическом пространстве текстов статей. "
            "Полезно, когда фото передаёт концепт или сцену, а не уникальный предмет."
        ),
    },
}

LOGO_SVG = """
<svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="52" height="52" rx="14" fill="#1a1a2e"/>
  <circle cx="22" cy="22" r="11" stroke="#e2e8f0" stroke-width="2.5" fill="none"/>
  <circle cx="22" cy="22" r="6"  stroke="#6366f1" stroke-width="2"   fill="none"/>
  <line x1="30" y1="30" x2="41" y2="41" stroke="#e2e8f0" stroke-width="3" stroke-linecap="round"/>
  <text x="22" y="26" text-anchor="middle" font-family="Georgia,serif"
        font-size="9" font-weight="bold" fill="#6366f1">W</text>
</svg>"""

DRAG_DROP_JS = """<script>
(function() {
  if (window._wl_dnd) return; window._wl_dnd = true;

  const ov = document.createElement('div');
  ov.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;gap:12px">
      <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
        <rect width="64" height="64" rx="18" fill="rgba(99,102,241,0.2)"/>
        <path d="M32 14v24M22 28l10 10 10-10" stroke="#6366f1" stroke-width="3"
              stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="14" y="42" width="36" height="8" rx="4" fill="#6366f1" opacity="0.4"/>
      </svg>
      <span style="font-size:18px;font-weight:600;color:#6366f1">Drop photo anywhere</span>
    </div>`;
  Object.assign(ov.style, {
    display:'none', position:'fixed', inset:'0',
    background:'rgba(10,10,30,0.55)',
    backdropFilter:'blur(4px)',
    zIndex:'9999',
    alignItems:'center', justifyContent:'center',
    pointerEvents:'none',
    transition:'opacity 0.15s',
  });
  document.body.appendChild(ov);

  let d = 0;
  document.addEventListener('dragenter', e => {
    if (!e.dataTransfer.types.includes('Files')) return;
    if (++d === 1) ov.style.display = 'flex';
  });
  document.addEventListener('dragleave', () => {
    if (--d <= 0) { d = 0; ov.style.display = 'none'; }
  });
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => {
    d = 0; ov.style.display = 'none'; e.preventDefault();
    const files = e.dataTransfer.files;
    if (!files?.length) return;
    const inp = document.querySelector('input[type="file"][accept]');
    if (!inp) return;
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    inp.files = dt.files;
    inp.dispatchEvent(new Event('change', { bubbles: true }));
  });
})();
</script>"""


@st.cache_data
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if sys.platform == "darwin":
        return "cpu"
    return "cpu"


CACHE_DIR = Path("/tmp/wikilens")

_CHECKPOINT_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def _download_checkpoint_files(hf_repo: str, checkpoint: str, is_lora: bool) -> Path:
    target = CACHE_DIR / checkpoint
    target.mkdir(parents=True, exist_ok=True)
    for fname in _CHECKPOINT_FILES:
        try:
            hf_hub_download(
                repo_id=hf_repo,
                filename=f"{checkpoint}/{fname}",
                repo_type="model",
                local_dir=str(CACHE_DIR),
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception:
            pass
    return target


def _hf_download_file(hf_repo: str, remote_path: str) -> Path:
    token = os.environ.get("HF_TOKEN")
    local = hf_hub_download(
        repo_id=hf_repo,
        filename=remote_path,
        repo_type="model",
        local_dir=str(CACHE_DIR),
        token=token,
    )
    return Path(local)


@st.cache_resource(show_spinner=False)
def _load_model(checkpoint: str, is_lora: bool, processor_id: str, hf_repo: str):
    device = _get_device()

    if not checkpoint:
        model = CLIPModel.from_pretrained(processor_id)
        return (
            model.to(device).eval(),  # type: ignore
            CLIPProcessor.from_pretrained(processor_id),
            device,
        )

    _WEIGHT_FILES = {
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    }

    def _is_valid_checkpoint(p: Path) -> bool:
        if not p.is_dir():
            return False
        return any((p / f).exists() for f in _WEIGHT_FILES)

    candidate = None
    for path in [Path(checkpoint), CACHE_DIR / checkpoint]:
        if _is_valid_checkpoint(path):
            candidate = path
            break

    if candidate is None:
        if hf_repo:
            candidate = _download_checkpoint_files(hf_repo, checkpoint, is_lora)
        else:
            raise ValueError(f"Checkpoint not found locally: {checkpoint!r}")

    if candidate is None or not candidate.is_dir():
        raise ValueError(f"Checkpoint not found: {checkpoint!r}")

    actual_is_lora = (candidate / "adapter_config.json").exists()

    if actual_is_lora:
        base = CLIPModel.from_pretrained(processor_id, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, str(candidate), is_trainable=False)
        model = model.merge_and_unload()  # type: ignore
        model = model.float()
    else:
        model = CLIPModel.from_pretrained(str(candidate), torch_dtype=torch.float32)

    return model.to(device).eval(), CLIPProcessor.from_pretrained(processor_id), device  # type: ignore


@st.cache_resource(show_spinner=False, ttl=INDEX_REFRESH_TTL)
def _load_index(faiss_path: str, meta_path: str, hf_repo: str):
    local_faiss = None
    for candidate in [Path(faiss_path), CACHE_DIR / faiss_path]:
        if candidate.is_file():
            local_faiss = candidate
            break
    if local_faiss is None and hf_repo:
        local_faiss = _hf_download_file(hf_repo, faiss_path)

    with open(local_faiss, "rb") as f:  # type: ignore
        index = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
    faiss.omp_set_num_threads(1)

    local_meta = None
    for candidate in [Path(meta_path), CACHE_DIR / meta_path]:
        if candidate.is_file():
            local_meta = candidate
            break

    if local_meta is None and hf_repo:
        try:
            local_meta = _hf_download_file(hf_repo, meta_path)
        except Exception as orig_e:
            fallback_path = None
            if meta_path.endswith(".json"):
                fallback_path = meta_path + "l"
            elif meta_path.endswith(".jsonl"):
                fallback_path = meta_path[:-1]

            if fallback_path:
                try:
                    local_meta = _hf_download_file(hf_repo, fallback_path)
                except Exception:
                    raise orig_e
            else:
                raise orig_e

    def _parse_meta_file(path: Path) -> dict:
        result: dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and raw:
                return raw
        except (json.JSONDecodeError, ValueError):
            pass

        valid_idx = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "title" in obj:
                        result[str(valid_idx)] = obj
                        valid_idx += 1
                except json.JSONDecodeError:
                    pass
        return result

    metadata = _parse_meta_file(local_meta)  # type: ignore

    if len(metadata) < 5000:
        meta_dir = local_meta.parent  # type: ignore
        for alt_name in ["metadata.json", "metadata.jsonl"]:
            alt_local = meta_dir / alt_name
            if alt_local == local_meta or not alt_local.exists():
                alt_remote = str(Path(meta_path).parent / alt_name)
                if hf_repo:
                    try:
                        alt_local = _hf_download_file(hf_repo, alt_remote)
                    except Exception:
                        continue
                else:
                    continue
            alt_meta = _parse_meta_file(alt_local)
            if len(alt_meta) > len(metadata):
                metadata = alt_meta
                break

    return index, metadata


def load_backend(backend: dict, processor_id: str, hf_repo: str):
    """
    Returns: model, processor, device, index_img, index_text, metadata_img, metadata_text
    index_text / metadata_text are None/{} if faiss_index_text is not set or fails.

    NOTE: image and text indices are built independently and have different integer IDs,
    so metadata must be loaded separately for each index.
    """
    try:
        model, processor, device = _load_model(
            backend["checkpoint"], backend["is_lora"], processor_id, hf_repo
        )
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}") from e

    try:
        index_img, metadata_img = _load_index(
            backend["faiss_index"], backend["metadata"], hf_repo
        )
    except Exception as e:
        idx_path = backend["faiss_index"]
        raise RuntimeError(
            f"Index load failed for '{idx_path}': {e}. "
            f"Check that this path exists in HF repo '{hf_repo}' and update backends.yaml."
        ) from e

    index_text = None
    metadata_text: dict = {}
    text_idx_path = backend.get("faiss_index_text", "")
    text_meta_path = backend.get("metadata_text", "")
    if not text_meta_path and text_idx_path:
        text_meta_path = str(Path(text_idx_path).parent / "metadata.jsonl")

    if text_idx_path:
        try:
            index_text, metadata_text = _load_index(
                text_idx_path, text_meta_path, hf_repo
            )
        except Exception:
            index_text = None
            metadata_text = {}

    return model, processor, device, index_img, index_text, metadata_img, metadata_text


def preload_all_backends(
    backends: list, processor_id: str, hf_repo: str, t: dict
) -> None:
    if st.session_state.get("_preload_done"):
        return

    ph = st.empty()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, b in enumerate(backends, 1):
        ph.info(
            t["warming"].format(i=i, n=len(backends)) + f" {b['emoji']} {b['label']}"
        )
        try:
            load_backend(b, processor_id, hf_repo)
            st.session_state.setdefault("warmed_backends", set()).add(b["id"])
        except Exception as e:
            ph.warning(f"⚠️ {b['label']}: {e}")
    ph.empty()
    st.session_state["_preload_done"] = True


def embed_image(image: Image.Image, model, processor, device: str) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return F.normalize(features, p=2, dim=-1).cpu().numpy().astype("float32")


def search(query_vec: np.ndarray, index: Any, metadata: dict, top_k: int) -> list:
    """Search a single index. Returns results with score in [0, 1]."""
    if index.ntotal == 0:
        return []

    fetch_k = min(top_k * 30, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)

    seen_titles: set = set()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = metadata.get(str(idx)) or metadata.get(str(int(idx))) or {}
        if not entry:
            continue
        title = entry.get("title", "")
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        results.append(
            {
                "title": title,
                "text": entry.get("text", ""),
                "image_path": entry.get("image_path", ""),
                "score": float(score),
            }
        )
        if len(results) >= top_k:
            break

    return results


def _ranked_entries(
    query_vec: np.ndarray, index: Any, metadata: dict, fetch_k: int
) -> list[tuple[str, dict]]:
    """
    Query one index; return (title, entry) pairs in rank order,
    deduplicated by title. Used by RRF and combined search.
    """
    scores, indices = index.search(query_vec, min(fetch_k, index.ntotal))
    seen: set = set()
    result = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = metadata.get(str(idx)) or metadata.get(str(int(idx))) or {}
        title = entry.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        entry = dict(entry)
        entry["_score"] = float(score)
        result.append((title, entry))
    return result


def search_rrf(
    query_vec: np.ndarray,
    index_img: Any,
    metadata_img: dict,
    index_text: Any,
    metadata_text: dict,
    top_k: int,
    k: int = 60,
) -> list:
    """
    Reciprocal Rank Fusion.

    Merges results by ARTICLE TITLE — this is critical because image and text
    indices are built independently and their integer IDs are unrelated.

    RRF score(d) = Σ_i  1 / (k + rank_i(d))
    """
    fetch_k = top_k * 30
    rrf_scores: dict[str, float] = {}
    title_to_entry: dict[str, dict] = {}

    for rank, (title, entry) in enumerate(
        _ranked_entries(query_vec, index_img, metadata_img, fetch_k)
    ):
        rrf_scores[title] = rrf_scores.get(title, 0.0) + 1.0 / (k + rank + 1)
        title_to_entry.setdefault(title, entry)

    n_indices = 1
    if index_text is not None:
        n_indices = 2
        for rank, (title, entry) in enumerate(
            _ranked_entries(query_vec, index_text, metadata_text, fetch_k)
        ):
            rrf_scores[title] = rrf_scores.get(title, 0.0) + 1.0 / (k + rank + 1)
            title_to_entry.setdefault(title, entry)

    max_rrf = n_indices / (k + 1)

    results = []
    for title, rrf_score in sorted(
        rrf_scores.items(), key=lambda x: x[1], reverse=True
    ):
        entry = title_to_entry[title]
        results.append(
            {
                "title": title,
                "text": entry.get("text", ""),
                "image_path": entry.get("image_path", ""),
                "score": min(rrf_score / max_rrf, 1.0),
            }
        )
        if len(results) >= top_k:
            break

    return results


def search_combined(
    query_vec: np.ndarray,
    index_img: Any,
    metadata_img: dict,
    index_text: Any,
    metadata_text: dict,
    top_k: int,
) -> list:
    """
    Score-based merge: keep the highest cosine similarity per article title
    across both indices.
    """
    fetch_k = top_k * 30
    best: dict[str, dict] = {}

    for title, entry in _ranked_entries(query_vec, index_img, metadata_img, fetch_k):
        score = entry["_score"]
        if title not in best or score > best[title]["score"]:
            best[title] = {
                "title": title,
                "text": entry.get("text", ""),
                "image_path": entry.get("image_path", ""),
                "score": score,
            }

    if index_text is not None:
        for title, entry in _ranked_entries(
            query_vec, index_text, metadata_text, fetch_k
        ):
            score = entry["_score"]
            if title not in best or score > best[title]["score"]:
                best[title] = {
                    "title": title,
                    "text": entry.get("text", ""),
                    "image_path": entry.get("image_path", ""),
                    "score": score,
                }

    return sorted(best.values(), key=lambda r: r["score"], reverse=True)[:top_k]


def search_dispatch(
    query_vec: np.ndarray,
    index_img: Any,
    metadata_img: dict,
    index_text: Any,
    metadata_text: dict,
    top_k: int,
    fusion_mode: str,
) -> list:
    if fusion_mode == "image_only" or index_text is None:
        return search(query_vec, index_img, metadata_img, top_k)
    elif fusion_mode == "text_only":
        return search(query_vec, index_text, metadata_text, top_k)
    elif fusion_mode == "combined":
        return search_combined(
            query_vec, index_img, metadata_img, index_text, metadata_text, top_k
        )
    else:
        return search_rrf(
            query_vec, index_img, metadata_img, index_text, metadata_text, top_k
        )


_THUMB_CACHE: dict = {}


def _fetch_one_thumb(title: str) -> str | None:
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(
            title.replace(" ", "_")
        )
        r = requests.get(url, timeout=6, headers={"User-Agent": "WikiLens/2.0"})
        if r.status_code == 200:
            src = r.json().get("thumbnail", {}).get("source")
            if src:
                return src
    except Exception:
        pass
    return None


def fetch_thumbnails(titles: list) -> dict:
    result = {t: _THUMB_CACHE[t] for t in titles if t in _THUMB_CACHE}
    to_fetch = [t for t in titles if t not in _THUMB_CACHE]

    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(len(to_fetch), 8)) as ex:
            futs = {ex.submit(_fetch_one_thumb, t): t for t in to_fetch}
            for fut in as_completed(futs):
                t = futs[fut]
                url = fut.result()
                if url:
                    _THUMB_CACHE[t] = url
                result[t] = url

    return result


def resolve_image(raw: str, base_dir: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    fb = Path(base_dir) / p.name
    return fb if fb.is_file() else None


def render_result(
    rank: int, result: dict, base_dir: str, t: dict, thumb_url: str | None
) -> None:
    with st.container(border=True):
        col_img, col_text = st.columns([1, 2], gap="medium")

        with col_img:
            if thumb_url:
                st.image(thumb_url, width="stretch")
            else:
                local_img = resolve_image(result["image_path"], base_dir)
                if local_img:
                    st.image(str(local_img), width="stretch")
                else:
                    st.markdown(
                        f"<div style='height:110px;display:flex;align-items:center;"
                        f"justify-content:center;background:#f5f5f5;border-radius:8px;"
                        f"color:#bbb;font-size:12px'>{t['no_image']}</div>",
                        unsafe_allow_html=True,
                    )

        with col_text:
            pct = result["score"] * 100
            color = "#2e7d32" if pct >= 60 else "#e65100" if pct < 35 else "#1565c0"
            st.markdown(
                f"<span style='font-size:11px;color:{color};font-weight:700;"
                f"letter-spacing:.05em;text-transform:uppercase'>"
                f"#{rank} &nbsp;·&nbsp; {pct:.1f}% {t['match']}</span>",
                unsafe_allow_html=True,
            )
            url = f"https://en.wikipedia.org/wiki/{quote(result['title'].replace(' ', '_'))}"
            st.markdown(f"### [{result['title']}]({url})")
            preview = result["text"][:320].rstrip()
            if len(result["text"]) > 320:
                preview += "…"
            st.write(preview)
            st.markdown(
                f"<a href='{url}' target='_blank' style='font-size:13px'>{t['read_more']}</a>",
                unsafe_allow_html=True,
            )


def render_sidebar(backends: list, t: dict, lang: str) -> tuple:
    """Returns (top_k, min_score)."""
    with st.sidebar:
        st.markdown(f"### {t['sidebar_title']}")
        other = "en" if lang == "ru" else "ru"
        lbl = "🇬🇧 EN" if lang == "ru" else "🇷🇺 RU"
        if st.button(lbl, key="lang_toggle", use_container_width=False):
            st.session_state["lang"] = other
            st.rerun()

        st.markdown("---")
        mode = st.radio(
            t["filter_label"],
            ["top_k", "threshold"],
            format_func=lambda x: (
                t["filter_top"] if x == "top_k" else t["filter_thresh"]
            ),
            horizontal=True,
        )
        if mode == "top_k":
            top_k, min_score = st.slider(t["topk_label"], 1, 5, 3), 0.0
        else:
            top_k = 5
            min_score = (
                st.slider(t["thresh_label"], 10, 90, 40, 5, help=t["thresh_help"])
                / 100.0
            )

        st.markdown("---")
        with st.expander(t["models_expander"], expanded=False):
            for b in backends:
                st.markdown(f"**{b['emoji']} {b['label']}**")
                desc = (
                    b.get("description_en", b["description"])
                    if lang == "en"
                    else b["description"]
                )
                st.caption(desc)

        st.markdown("---")
        with st.expander(t["index_expander"], expanded=False):
            for key in (
                "idx_desc_rrf",
                "idx_desc_combined",
                "idx_desc_image",
                "idx_desc_text",
            ):
                st.markdown(t[key])
                st.markdown("")

    return top_k, min_score


_POPOVER_STYLE = """<style>
div[data-testid="stPopover"] div[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
</style>"""

_FUSION_OPTIONS = ["rrf", "combined", "image_only", "text_only"]


def render_model_picker(backends: list, t: dict, lang: str) -> dict:
    if "selected_backend_idx" not in st.session_state:
        st.session_state["selected_backend_idx"] = 4
    idx = st.session_state["selected_backend_idx"]
    current = backends[idx]

    st.markdown(_POPOVER_STYLE, unsafe_allow_html=True)

    with st.popover(
        f"{current['emoji']} {current['label']} ▾", use_container_width=False
    ):
        for i, b in enumerate(backends):
            prefix = "✓ " if i == idx else "   "
            if st.button(
                f"{prefix}{b['emoji']} {b['label']}",
                key=f"mbtn_{i}",
                use_container_width=True,
            ):
                st.session_state["selected_backend_idx"] = i
                st.rerun()

    return backends[st.session_state["selected_backend_idx"]]


def render_index_picker(t: dict) -> str:
    if "selected_fusion_idx" not in st.session_state:
        st.session_state["selected_fusion_idx"] = 0

    idx = st.session_state["selected_fusion_idx"]
    _labels = {
        "rrf": t["fusion_rrf"],
        "combined": t["fusion_combined"],
        "image_only": t["fusion_image"],
        "text_only": t["fusion_text"],
    }
    current_label = _labels[_FUSION_OPTIONS[idx]]

    with st.popover(f"{current_label} ▾", use_container_width=False):
        for i, mode in enumerate(_FUSION_OPTIONS):
            prefix = "✓ " if i == idx else "   "
            if st.button(
                f"{prefix}{_labels[mode]}",
                key=f"fbtn_{i}",
                use_container_width=True,
            ):
                st.session_state["selected_fusion_idx"] = i
                st.rerun()

    return _FUSION_OPTIONS[st.session_state["selected_fusion_idx"]]


CONTACTS_HTML = """
<div style='display:flex;gap:20px;align-items:center;flex-wrap:wrap;justify-content:center'>
  <a href='https://github.com/justnurik' target='_blank'
     style='text-decoration:none;color:inherit;display:flex;align-items:center;gap:6px;font-size:14px;opacity:.7'>
    <svg width='18' height='18' viewBox='0 0 24 24' fill='currentColor'><path d='M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z'/></svg>
    GitHub
  </a>
  <a href='https://huggingface.co/jnurik' target='_blank'
     style='text-decoration:none;color:inherit;display:flex;align-items:center;gap:6px;font-size:14px;opacity:.7'>
    🤗 HuggingFace
  </a>
  <a href='https://t.me/jnurik' target='_blank'
     style='text-decoration:none;color:inherit;display:flex;align-items:center;gap:6px;font-size:14px;opacity:.7'>
    <svg width='16' height='16' viewBox='0 0 24 24' fill='currentColor'><path d='M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.894 8.221-1.97 9.28c-.145.658-.537.818-1.084.508l-3-2.21-1.447 1.394c-.16.16-.295.295-.605.295l.213-3.053 5.56-5.023c.242-.213-.054-.333-.373-.12L7.19 13.53l-2.965-.924c-.643-.204-.657-.643.136-.953l11.57-4.461c.537-.194 1.006.131.963.03z'/></svg>
    Telegram
  </a>
</div>"""


def render_footer(t: dict) -> None:
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;color:#888;font-size:13px;margin-bottom:6px'>"
        f"{t['contacts_title']}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(CONTACTS_HTML, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def main():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ru"
    lang = st.session_state["lang"]
    t = T[lang]

    st.set_page_config(page_title=t["page_title"], page_icon="🔍", layout="wide")
    st.html(DRAG_DROP_JS)

    try:
        cfg = load_config(BACKENDS_CONFIG_PATH)
    except FileNotFoundError:
        st.error(t["config_error"].format(path=BACKENDS_CONFIG_PATH))
        st.stop()

    backends = cfg["backends"]
    processor_id = cfg["processor_id"]
    images_base_dir = cfg.get("images_base_dir", ".")
    hf_repo = cfg.get("hf_repo", "")

    if "warmed_backends" not in st.session_state:
        st.session_state["warmed_backends"] = set()

    preload_all_backends(backends, processor_id, hf_repo, t)

    top_k, min_score = render_sidebar(backends, t, lang)

    col_logo, col_title = st.columns([0.5, 9], gap="small")
    with col_logo:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
    with col_title:
        st.markdown(
            "<h1 style='margin:0;padding:4px 0 0;font-size:2rem'>WikiLens</h1>",
            unsafe_allow_html=True,
        )

    st.markdown(t["hero_desc"])

    exp_col1, exp_col2 = st.columns(2, gap="small")
    with exp_col1:
        with st.expander(t["model_disclaimer_title"], expanded=False):
            st.markdown(t["model_disclaimer"])
    with exp_col2:
        with st.expander(t["index_disclaimer_title"], expanded=False):
            st.markdown(t["index_disclaimer"])

    st.divider()

    col_input, col_results = st.columns([1, 2.5], gap="large")

    with col_input:
        st.markdown(t["upload_label"])
        uploaded_file = st.file_uploader(
            "photo",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        st.caption(t["upload_caption"])

        st.markdown(t["model_label"])
        selected_backend = render_model_picker(backends, t, lang)

        st.markdown(t["index_label"])
        fusion_mode = render_index_picker(t)

        if uploaded_file is not None:
            try:
                query_image = ImageOps.exif_transpose(
                    Image.open(uploaded_file)
                ).convert("RGB")
                st.image(query_image, width="stretch", caption="")
            except Exception as e:
                st.error(t["open_error"].format(e=e))
                uploaded_file = None

    with col_results:
        if uploaded_file is None:
            st.markdown(
                f"<div style='padding:40px 0;text-align:center;color:#666;font-size:16px'>"
                f"{t['placeholder']}</div>",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner(t["searching"]):
                try:
                    (
                        model,
                        processor,
                        device,
                        index_img,
                        index_text,
                        metadata_img,
                        metadata_text,
                    ) = load_backend(selected_backend, processor_id, hf_repo)
                except Exception as e:
                    st.error(t["model_error"].format(e=e))
                    st.stop()

                if index_text is None and fusion_mode in (
                    "rrf",
                    "combined",
                    "text_only",
                ):
                    st.warning(t["fusion_no_text_index"])

                try:
                    query_vec = embed_image(query_image, model, processor, device)  # type: ignore
                    results = search_dispatch(
                        query_vec,
                        index_img,
                        metadata_img,
                        index_text,
                        metadata_text,
                        top_k=top_k,
                        fusion_mode=fusion_mode,
                    )
                except Exception as e:
                    st.error(t["search_error"].format(e=e))
                    st.stop()

            if min_score > 0:
                results = [r for r in results if r["score"] >= min_score]

            if not results:
                st.warning(
                    t["no_results"].format(pct=int(min_score * 100))
                    if min_score > 0
                    else f"No results. Index: {index_img.ntotal} vectors, metadata: {len(metadata_img)}."
                )
            else:
                if len(metadata_img) < index_img.ntotal * 0.5:
                    st.warning(
                        f"⚠️ Metadata mismatch: индекс {index_img.ntotal:,} векторов, "
                        f"metadata {len(metadata_img):,} записей. "
                        f"Пересоберите индекс командой `python build_index.py`."
                    )

                thumbnails = fetch_thumbnails([r["title"] for r in results])

                st.session_state["last_results"] = results
                st.session_state["last_thumbnails"] = thumbnails
                st.session_state["last_base_dir"] = images_base_dir

                st.markdown(t["results_title"])
                for rank, r in enumerate(results, 1):
                    render_result(
                        rank, r, images_base_dir, t, thumbnails.get(r["title"])
                    )

    render_footer(t)


if __name__ == "__main__":
    main()
