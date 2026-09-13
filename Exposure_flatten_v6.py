#!/usr/bin/env python3
"""
exposure_flatten.py — математическое выравнивание экспозиции по алгоритму
"flat-field correction" (перенос из астро/научной фотографии в ретушь).

Алгоритм:
    L(x,y)     — яркость (Lab L-канал) исходного изображения
    B(x,y)     = GaussianBlur(L, radius)      — карта базовой засветки (низкие частоты)
    target     = mean(L) по всему изображению — глобальное среднее
    correction = (target - B(x,y)) * strength — сила поправки в каждой точке
    L'(x,y)    = clip(L(x,y) + correction, 0, 100)

Ключевое свойство: поправка вычисляется из РАЗМЫТОЙ карты, а не из
оригинала — поэтому текстура (поры кожи, ткань, микроконтраст) не
трогается, меняется только общий тональный рельеф ("свет/тень крупных форм").

Требования: pip install numpy pillow scipy --break-system-packages
Использование:
    python3 exposure_flatten.py input.jpg
    python3 exposure_flatten.py input.jpg --radius 120 --strength 0.5
    python3 exposure_flatten.py input.jpg --points "1200,800;300,900"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB (0-255, uint8) -> Lab (L: 0-100, a/b: ~-128..127). Pure numpy, no deps beyond numpy."""
    rgb = rgb_uint8.astype(np.float64) / 255.0

    # sRGB -> linear RGB
    mask = rgb > 0.04045
    linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # linear RGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T

    # normalize by D65 white point
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])

    # XYZ -> Lab
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Lab -> RGB (0-255, uint8). Inverse of rgb_to_lab."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    delta = 6 / 29

    def finv(t):
        return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    xyz = xyz * np.array([0.95047, 1.00000, 1.08883])

    M_inv = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    linear = xyz @ M_inv.T
    linear = np.clip(linear, 0, None)

    mask = linear > 0.0031308
    srgb = np.where(mask, 1.055 * np.power(linear, 1 / 2.4) - 0.055, 12.92 * linear)
    srgb = np.clip(srgb, 0, 1)

    return (srgb * 255).astype(np.uint8)


def robust_target(L: np.ndarray, mode: str = "trimmed", trim_percent: float = 5.0,
                   manual: float | None = None) -> float:
    """
    Умное определение 'мишени' (target) вместо наивного mean(L).

    Проблема наивного mean: его легко утаскивает случайный яркий блик,
    тёмная рамка кадра, чёрный паспарту вокруг картины и т.п. — то есть
    небольшая по площади, но экстремальная по яркости зона искажает
    среднее для ВСЕГО кадра.

    mode:
        'mean'    — арифметическое среднее (старое поведение, для сравнения)
        'median'  — медиана: устойчива к выбросам, но менее repräsentative
                    для плавных градиентов
        'trimmed' — обрезанное среднее: убираем trim_percent% самых тёмных
                    и trim_percent% самых светлых пикселей, затем усредняем
                    остаток. Лучший баланс: устойчиво к выбросам (рамка,
                    блик), но чувствительно к форме основного распределения.
                    Рекомендуется по умолчанию.
        'mode'    — пик гистограммы (самый частый тон) — то, что глаз
                    воспринимает как "основной тон" сцены. Хорошо для
                    сцен с одним доминирующим полем (стена, холст, фон).
    manual: если задано (0-100) — просто используется как target напрямую,
        игнорируя автоматику. Даёт фотографу художественный контроль,
        когда автоматика "не понимает" сцену (нарочито тёмный/светлый кадр).
    """
    if manual is not None:
        return float(manual)

    if mode == "mean":
        return float(L.mean())

    if mode == "median":
        return float(np.median(L))

    if mode == "trimmed":
        lo, hi = np.percentile(L, [trim_percent, 100 - trim_percent])
        window = L[(L >= lo) & (L <= hi)]
        if window.size == 0:
            return float(L.mean())
        return float(window.mean())

    if mode == "mode":
        hist, edges = np.histogram(L, bins=100, range=(0, 100))
        # лёгкое сглаживание гистограммы, чтобы не попасть в случайный шумный пик
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        hist_smooth = np.convolve(hist, kernel, mode="same")
        peak_bin = int(np.argmax(hist_smooth))
        return float((edges[peak_bin] + edges[peak_bin + 1]) / 2)

    raise ValueError(f"Неизвестный target_mode: {mode}")


def target_extremity(target: float) -> float:
    """0.0 в середине (target=50), 1.0 на крайних значениях (target=0 или 100)."""
    return min(abs(target - 50.0) / 50.0, 1.0)


def soft_clip(x: np.ndarray, lo: float = 0.0, hi: float = 100.0, knee: float = 6.0) -> np.ndarray:
    """
    Мягкое ограничение диапазона вместо жёсткого np.clip.

    Жёсткий clip создаёт резкий обрыв на границе 0/100 — плоский "срез"
    без деталей (banding, потеря пластики в самых тёмных/светлых зонах).
    soft_clip вместо обрыва делает экспоненциальный "подход" к границе —
    похоже на характеристическую кривую плёнки (toe/shoulder): тени не
    проваливаются в чистый чёрный резко, света не обрезаются в чистый белый.

    knee — ширина зоны смягчения в пунктах L. Чем больше knee, тем раньше
    начинается сжатие и тем мягче переход (более "плёночный" характер).
    """
    x = x.astype(np.float64)
    out = x.copy()

    hi_start = hi - knee
    mask_hi = x > hi_start
    excess = x[mask_hi] - hi_start
    out[mask_hi] = hi_start + knee * (1 - np.exp(-excess / knee))

    lo_start = lo + knee
    mask_lo = x < lo_start
    deficit = lo_start - x[mask_lo]
    out[mask_lo] = lo_start - knee * (1 - np.exp(-deficit / knee))

    return np.clip(out, lo, hi)  # финальная страховка от математических хвостов


def apply_film_look(lab: np.ndarray, lift: float, rolloff: float,
                     warmth: float, grain: float, rng: np.random.Generator) -> np.ndarray:
    """
    Плёночный вайб поверх выровненного изображения. Работает после основной
    коррекции экспозиции, отдельным проходом.

    lift    (0-15, обычно 3-6): поднимает чёрную точку — тени никогда не
            уходят в чистый 0, а имеют лёгкую "дымку", как непроявленная
            база плёнки. L' = lift + L*(100-lift)/100
    rolloff (0-25, обычно 10-15): мягкость сжатия светов через soft_clip
            с увеличенным knee — света "заворачивают" плавно, а не режутся.
    warmth  (0-1, обычно 0.2-0.4): split-toning — тени чуть теплее
            (сдвиг Lab b+ / a+), света чуть холоднее (b- / a-), классика
            цветной плёнки.
    grain   (0-1, обычно 0.1-0.25): лёгкий шум по L-каналу, сильнее в
            тенях слабее в светах (как плёночное зерно, которое заметнее
            в недоэкспонированных участках).
    """
    lab = lab.copy()
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # 1. Lift теней (черная точка приподнята)
    if lift > 0:
        L = lift + L * (100 - lift) / 100.0

    # 2. Мягкий rolloff светов (использует уже определённую soft_clip с увеличенным knee)
    if rolloff > 0:
        L = soft_clip(L, lo=0, hi=100, knee=rolloff)

    # 3. Split-toning: тепло в тенях, прохлада в светах
    if warmth > 0:
        shadow_weight = np.clip((50 - L) / 50, 0, 1)   # 1 в чёрном, 0 в среднем сером
        light_weight = np.clip((L - 50) / 50, 0, 1)    # 1 в белом, 0 в среднем сером
        a = a + warmth * (shadow_weight * 4 - light_weight * 2)   # тени -> чуть в красный
        b = b + warmth * (shadow_weight * 8 - light_weight * 6)   # тени -> в жёлтый, света -> в синий

    # 4. Зерно: сильнее в тенях, слабее в светах
    if grain > 0:
        grain_weight = np.clip((70 - L) / 70, 0.15, 1.0)  # даже в светах остаётся немного
        noise = rng.normal(0, grain * 6.0, L.shape) * grain_weight
        L = L + noise

    lab[..., 0] = np.clip(L, 0, 100)
    lab[..., 1] = a
    lab[..., 2] = b
    return lab


def detect_content_mask(lab: np.ndarray, border_frac: float = 0.03,
                         sensitivity: float = 4.0) -> np.ndarray:
    """
    Автоматическое отделение 'объекта' (картина) от 'окружения' (рама, паспарту, стена).

    Логика: берём тонкую полосу по периметру кадра (border_frac от короткой стороны) —
    считаем, что это окружение, а не сам объект (обычно верно для съёмки картин).
    Меряем средний цвет/яркость этой полосы в Lab (L,a,b — учитывает и тон, и цвет,
    не только яркость, так что серая/цветная рама тоже ловится).
    Дальше для каждого пикселя считаем расстояние до этого 'цвета окружения'.
    Пиксели, которые СИЛЬНО отличаются от типичной окраски окружения — это объект.
    Порог = mean(dist по границе) + sensitivity * std(dist по границе) — то есть
    порог адаптивный, а не фиксированное число, подстраивается под то, насколько
    сама рама неоднородна (шум, лёгкий градиент на паспарту и т.п.).

    Возвращает soft-маску 0..1 (1 = точно объект, 0 = точно окружение).
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    h, w = L.shape
    bw = max(2, int(min(h, w) * border_frac))

    border_L = np.concatenate([L[:bw, :].ravel(), L[-bw:, :].ravel(), L[:, :bw].ravel(), L[:, -bw:].ravel()])
    border_a = np.concatenate([a[:bw, :].ravel(), a[-bw:, :].ravel(), a[:, :bw].ravel(), a[:, -bw:].ravel()])
    border_b = np.concatenate([b[:bw, :].ravel(), b[-bw:, :].ravel(), b[:, :bw].ravel(), b[:, -bw:].ravel()])

    Ls, as_, bs = border_L.mean(), border_a.mean(), border_b.mean()
    border_dist = np.sqrt((border_L - Ls) ** 2 + (border_a - as_) ** 2 + (border_b - bs) ** 2)
    mu, sigma = border_dist.mean(), border_dist.std()
    threshold = mu + sensitivity * max(sigma, 0.5)  # защита от sigma≈0 на идеально ровной рамке

    dist = np.sqrt((L - Ls) ** 2 + (a - as_) ** 2 + (b - bs) ** 2)
    hard_mask = (dist > threshold).astype(np.float64)

    # чистим мелкий шум маски (убираем отдельные пиксели-выбросы внутри рамки/внутри объекта)
    hard_mask = gaussian_filter(hard_mask, sigma=max(2.0, min(h, w) * 0.004))
    hard_mask = (hard_mask > 0.5).astype(np.float64)

    return hard_mask


def build_roi_mask(shape: tuple[int, int], roi_rect: tuple[int, int, int, int] | None,
                    roi_percent: float | None) -> np.ndarray | None:
    """Строит прямоугольную маску по ручному ROI (пиксели или %). None, если ROI не задан."""
    h, w = shape
    if roi_rect is not None:
        x, y, rw, rh = roi_rect
        mask = np.zeros((h, w), dtype=np.float64)
        x2, y2 = min(w, x + rw), min(h, y + rh)
        mask[max(0, y):y2, max(0, x):x2] = 1.0
        return mask
    if roi_percent is not None:
        frac = np.clip(roi_percent / 100.0, 0.01, 1.0)
        margin_y = int(h * (1 - frac) / 2)
        margin_x = int(w * (1 - frac) / 2)
        mask = np.zeros((h, w), dtype=np.float64)
        mask[margin_y:h - margin_y, margin_x:w - margin_x] = 1.0
        return mask
    return None


def masked_gaussian_blur(L: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Размытие, нормализованное по маске (normalized convolution) — классический
    приём, чтобы яркие/тёмные пиксели ИЗ-ЗА ПРЕДЕЛОВ маски (окружение) не
    'протекали' в карту базовой засветки B(x,y) внутри маски (объект) рядом
    с границей. Без этого рядом с краем рамки образуется ореол/хало.
    """
    num = gaussian_filter(L * mask, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask, sigma=sigma, mode="nearest")
    return num / np.clip(den, 1e-6, None)


def enhance_volume(L: np.ndarray, mask_soft: np.ndarray | None,
                    radius: float, strength: float, max_boost: float | None) -> np.ndarray:
    """
    Автоматизированный Dodge & Burn для усиления объёма (не путать с выравниванием
    экспозиции — это ПРОТИВОПОЛОЖНАЯ по духу операция).

    Идея та же математика (локальное среднее через blur), но наоборот: вместо того
    чтобы СТИРАТЬ отклонение зоны от локального среднего, мы его УСИЛИВАЕМ.
    Каждый пиксель сравнивается со своим локальным окружением (сглаженной версией
    себя) — то, что чуть темнее своего окружения, становится ещё темнее (burn),
    то, что чуть светлее — ещё светлее (dodge). Это в точности имитирует ручной
    D&B на сером слое (soft light/overlay), только без кисти — по всему кадру сразу,
    с учётом формы объекта.

    detail(x,y) = L(x,y) - GaussianBlur(L(x,y), radius)   — "форма" без учёта детали
    L'(x,y)     = L(x,y) + detail(x,y) * strength          — усиливаем эту форму

    radius задаёт МАСШТАБ объёма, который усиливается:
        маленький radius (5-15px)   — микро-контраст, "хруст" текстуры/мазков
        средний radius (30-80px)    — объём отдельных форм (складки, световые пятна)
        большой radius (100px+)     — крупная лепка объёма всей композиции
    """
    local_avg = gaussian_filter(L, sigma=radius, mode="nearest")
    detail = L - local_avg
    boost = detail * strength
    if max_boost is not None:
        boost = np.clip(boost, -max_boost, max_boost)
    if mask_soft is not None:
        boost = boost * mask_soft
    return L + boost


def apply_halation(lab: np.ndarray, threshold: float, strength: float,
                    radius: float, tint_a: float, tint_b: float) -> np.ndarray:
    """
    Halation — мягкое тёплое свечение вокруг ярких участков (свет "просачивается"
    сквозь эмульсию плёнки на подложку и отражается обратно). Характерно для
    цветной плёночной фотографии, особенно на пересветах/бликах.

    Работает только с зонами ЯРЧЕ threshold — берёт их "силу свечения",
    размывает на radius (даёт мягкий ореол), подмешивает обратно как
    прибавку к яркости + тёплый цветовой сдвиг (a+/b+ — в красно-оранжевый).
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    highlight_mask = np.clip((L - threshold) / max(100 - threshold, 1e-6), 0, 1)
    glow = gaussian_filter(highlight_mask, sigma=radius, mode="nearest")

    lab_out = lab.copy()
    lab_out[..., 0] = np.clip(L + glow * strength * 25, 0, 100)
    lab_out[..., 1] = a + glow * strength * tint_a
    lab_out[..., 2] = b + glow * strength * tint_b
    return lab_out


def apply_vignette(lab: np.ndarray, strength: float, radius: float, softness: float) -> np.ndarray:
    """
    Плавное затемнение к углам кадра — оптическая характеристика объектива,
    часто ассоциируется с плёночными/винтажными камерами.

    radius: на каком относительном расстоянии от центра (0-1) начинается затемнение
    softness: ширина перехода (0-1) — больше = мягче/плавнее граница
    strength: во сколько раз темнее становятся самые углы (0-1)
    """
    L = lab[..., 0]
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2)
    dist = np.clip(dist, 0, 1)
    falloff = np.clip((dist - radius) / max(softness, 1e-6), 0, 1)
    factor = 1.0 - strength * falloff

    lab_out = lab.copy()
    lab_out[..., 0] = L * factor
    return lab_out


def flatten_exposure(image_path: str, radius: float, strength: float,
                      out_path: str, map_path: str,
                      sample_points: list[tuple[int, int]] | None,
                      mode: str = "both", threshold: float = 0.0,
                      max_correction: float | None = None, power: float = 1.0,
                      target_mode: str = "trimmed", trim_percent: float = 5.0,
                      manual_target: float | None = None, adaptive: bool = True,
                      clip_knee: float = 6.0,
                      film: bool = False, film_lift: float = 4.0,
                      film_rolloff: float = 12.0, film_warmth: float = 0.3,
                      film_grain: float = 0.15, seed: int = 0,
                      roi_rect: tuple[int, int, int, int] | None = None,
                      roi_percent: float | None = None,
                      auto_roi: bool = False, roi_sensitivity: float = 4.0,
                      roi_feather: float | None = None,
                      correction_scope: str = "roi",
                      dodge_burn: bool = False, db_radius: float = 60.0,
                      db_strength: float = 0.3, db_max: float | None = 15.0,
                      halation: bool = False, halation_threshold: float = 80.0,
                      halation_strength: float = 0.3, halation_radius: float = 15.0,
                      vignette: bool = False, vignette_strength: float = 0.25,
                      vignette_radius: float = 0.85, vignette_softness: float = 0.5):
    """
    mode: 'both' | 'darks' | 'lights'
        'darks'  — трогает только зоны темнее target (осветляет их), светлые не трогает
        'lights' — трогает только зоны светлее target (затемняет их), тёмные не трогает
        'both'   — оба направления (поведение по умолчанию)
    threshold: "мёртвая зона" в пунктах L. Зоны, чьё отклонение от target меньше
        threshold, не трогаются вообще — так мелкие, незначительные перепады не лезут
        под коррекцию, только реально "выбивающиеся" участки.
    max_correction: жёсткий потолок величины поправки в пунктах L (после strength).
        Защищает глубокие тени/яркие света от переисправления.
    power: степень чувствительности кривой поправки.
        power=1   — линейно, поправка пропорциональна отклонению (по умолчанию)
        power<1 (напр. 0.6) — агрессивнее на малых отклонениях, слабо растёт дальше
                              (подтягивает лёгкие несоответствия сильнее)
        power>1 (напр. 1.5) — мягче на малых отклонениях, сильнее только на больших
                              (не трогает лёгкие расхождения, бьёт только явный перепад)
    target_mode / trim_percent / manual_target: см. robust_target()
    adaptive: если True — автоматически ослабляет strength, когда target сам по себе
        экстремальный (очень тёмная или очень светлая сцена), чтобы не пытаться
        "вытянуть" низкий/высокий ключ в средний серый и не спровоцировать клиппинг.
    clip_knee: ширина мягкого ограничения диапазона (см. soft_clip). 0 = жёсткий clip.
    film / film_*: см. apply_film_look()
    roi_rect / roi_percent / auto_roi / roi_sensitivity / roi_feather: см. build_roi_mask() / detect_content_mask()
    correction_scope: 'roi' | 'full'
        'roi'  (по умолчанию) — и target, и сама коррекция ограничены областью объекта;
               рама/паспарту физически не трогаются.
        'full' — target вычисляется ТОЛЬКО по объекту (ROI), но сама коррекция
               применяется ко всему кадру целиком, включая раму/паспарту/фон.
    dodge_burn / db_radius / db_strength / db_max: см. enhance_volume() — усиление
        объёма (противоположность выравниванию: не гасит локальные отклонения, а
        усиливает их). Ограничивается той же ROI-маской, что и основная коррекция
        (при correction_scope='roi' — только объект; при 'full' — весь кадр).
    halation / halation_*: см. apply_halation() — плёночное свечение вокруг бликов.
    vignette / vignette_*: см. apply_vignette() — плавное затемнение к углам кадра.
    """

    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)

    print(f"[1/8] Изображение: {rgb.shape[1]}x{rgb.shape[0]}, конвертирую в Lab...")
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    h, w = L.shape

    # --- ROI: отделяем объект от окружения (рама/паспарту/стена), если задано ---
    content_mask = None
    if roi_rect is not None or roi_percent is not None:
        content_mask = build_roi_mask((h, w), roi_rect, roi_percent)
        print(f"      ROI: ручная область задана, {content_mask.mean()*100:.0f}% кадра считается объектом")
    if auto_roi:
        auto_mask = detect_content_mask(lab, sensitivity=roi_sensitivity)
        content_mask = auto_mask if content_mask is None else content_mask * auto_mask
        print(f"      ROI: авто-детект объекта по контрасту с рамкой, "
              f"{content_mask.mean()*100:.0f}% кадра определено как объект")

    if content_mask is not None:
        feather = roi_feather if roi_feather is not None else min(h, w) * 0.015
        content_mask_soft = gaussian_filter(content_mask, sigma=feather, mode="nearest")
        target_pixels = L[content_mask > 0.5]
        if target_pixels.size < 100:  # маска выродилась (слишком мало пикселей) — не доверяем ей
            print("      ROI: маска объекта слишком мала, откатываюсь к целому кадру")
            content_mask = None
            content_mask_soft = None
            target_pixels = L
    else:
        content_mask_soft = None
        target_pixels = L

    target = robust_target(target_pixels, mode=target_mode, trim_percent=trim_percent, manual=manual_target)
    naive_mean = float(L.mean())
    print(f"[2/8] Target (mode={target_mode}{'  на ROI' if content_mask is not None else ''}) L = {target:.2f}  "
          f"(для сравнения, наивный mean по всему кадру = {naive_mean:.2f})")

    strength_eff = strength
    if adaptive:
        extremity = target_extremity(target)
        if extremity > 0.3:  # target заметно отличается от среднего серого (50)
            damping = 1.0 - 0.5 * (extremity - 0.3) / 0.7   # плавно снижаем силу до 50% на самых крайних target
            strength_eff = strength * damping
            print(f"      Adaptive: target далёк от 50 (extremity={extremity:.2f}), "
                  f"strength снижена {strength:.2f} -> {strength_eff:.2f} "
                  f"(защита от пере-коррекции low-key/high-key сцены)")

    print(f"[3/8] Строю карту базовой засветки (Gaussian blur, radius={radius})...")
    if content_mask is not None and correction_scope == "roi":
        # masked/normalized blur — яркость окружения (рамы) не просачивается в B внутри объекта
        # (используется только когда и коррекция ограничена ROI — иначе смысла в masked-blur нет,
        # т.к. вне ROI мы всё равно хотим честную, неискажённую блюр-карту всего кадра)
        B = masked_gaussian_blur(L, content_mask, sigma=radius)
    else:
        B = gaussian_filter(L, sigma=radius, mode="nearest")  # nearest вместо reflect — без зеркального артефакта у краёв кадра

    diff = target - B  # >0 в тёмных зонах (нужно осветлить), <0 в светлых (нужно затемнить)

    # sensitivity curve (power)
    sign = np.sign(diff)
    magnitude = np.abs(diff)
    if power != 1.0:
        magnitude = magnitude ** power
    correction = sign * magnitude * strength_eff

    # мёртвая зона: убираем правку там, где исходное отклонение было мало
    if threshold > 0:
        correction = np.where(np.abs(diff) < threshold, 0.0, correction)

    # направленность: только тёмные / только светлые зоны
    if mode == "darks":
        correction = np.where(diff > 0, correction, 0.0)   # только осветление теней
    elif mode == "lights":
        correction = np.where(diff < 0, correction, 0.0)   # только затемнение светов

    # потолок силы поправки
    if max_correction is not None:
        correction = np.clip(correction, -max_correction, max_correction)

    # ROI: коррекция применяется только внутри объекта (с плавным затуханием к краю),
    # либо, если correction_scope='full' — target взят из ROI, но коррекция идёт по всему кадру
    if content_mask_soft is not None and correction_scope == "roi":
        correction = correction * content_mask_soft
    elif content_mask is not None and correction_scope == "full":
        print(f"      Correction scope: 'full' — target вычислен по ROI, но поправка "
              f"применяется ко всему кадру (включая раму/паспарту)")

    L_new = L + correction
    if clip_knee > 0:
        L_new = soft_clip(L_new, lo=0, hi=100, knee=clip_knee)
    else:
        L_new = np.clip(L_new, 0, 100)

    print(f"[4/8] Готово. ({'мягкое' if clip_knee > 0 else 'жёсткое'} ограничение диапазона)")

    lab_new = lab.copy()
    lab_new[..., 0] = L_new

    if dodge_burn:
        # тот же принцип ROI-ограничения, что и у основной коррекции экспозиции:
        # при correction_scope='roi' объём усиливается только внутри объекта,
        # при 'full' — по всему кадру
        db_mask = content_mask_soft if (content_mask_soft is not None and correction_scope == "roi") else None
        print(f"[5/8] Dodge & Burn: усиливаю объём (radius={db_radius}, strength={db_strength})...")
        L_new = enhance_volume(L_new, db_mask, db_radius, db_strength, db_max)
        lab_new[..., 0] = L_new
    else:
        print("[5/8] Dodge & Burn выключен (--dodge-burn не задан)")

    if halation:
        print(f"[6/8] Halation: добавляю плёночное свечение бликов "
              f"(threshold={halation_threshold}, strength={halation_strength})...")
        lab_new = apply_halation(lab_new, halation_threshold, halation_strength,
                                  halation_radius, tint_a=6.0, tint_b=10.0)
        L_new = lab_new[..., 0]
    else:
        print("[6/8] Halation выключен (--halation не задан)")

    if vignette:
        print(f"[7/8] Виньетирование (strength={vignette_strength}, radius={vignette_radius})...")
        lab_new = apply_vignette(lab_new, vignette_strength, vignette_radius, vignette_softness)
        L_new = lab_new[..., 0]
    else:
        print("[7/8] Виньетирование выключено (--vignette не задан)")

    if film:
        print(f"[8/8] Применяю плёночный тон-курв (lift={film_lift}, rolloff={film_rolloff}, "
              f"warmth={film_warmth}, grain={film_grain})...")
        rng = np.random.default_rng(seed)
        lab_new = apply_film_look(lab_new, film_lift, film_rolloff, film_warmth, film_grain, rng)
        L_new = lab_new[..., 0]  # обновляем L_new для диагностики ниже (после film-грейда)
    else:
        print("[8/8] Плёночный тон-курв выключен (--film не задан)")

    print("Конвертирую обратно в RGB и сохраняю...")
    rgb_new = lab_to_rgb(lab_new)
    Image.fromarray(rgb_new).save(out_path, quality=95)

    # диагностическая карта поправки: 128 = 0 правки, светлее = осветление, темнее = затемнение
    corr_vis = np.clip(128 + correction * 4, 0, 255).astype(np.uint8)
    Image.fromarray(corr_vis).save(map_path)

    print(f"      Сохранено: {out_path}")
    print(f"      Карта поправки: {map_path} (серый=без изменений, светлое=осветлено, тёмное=затемнено)")

    print("Точечная проверка:")
    if sample_points:
        print(f"      {'X':>6} {'Y':>6} {'L до':>8} {'L после':>8} {'Δ до':>8} {'Δ после':>8}")
        for (x, y) in sample_points:
            if 0 <= y < L.shape[0] and 0 <= x < L.shape[1]:
                l_before = L[y, x]
                l_after = L_new[y, x]
                print(f"      {x:>6} {y:>6} {l_before:>8.2f} {l_after:>8.2f} "
                      f"{target - l_before:>8.2f} {target - l_after:>8.2f}")
            else:
                print(f"      Точка ({x},{y}) вне границ изображения — пропущена")
    else:
        print("      (точки не заданы — используйте --points 'x1,y1;x2,y2' для проверки конкретных зон)")

    print(f"\n      Итоговое среднее L после коррекции: {L_new.mean():.2f} (было {target:.2f})")
    print(f"      Std L до: {L.std():.2f}  →  Std L после: {L_new.std():.2f}  "
          f"(меньше std = ровнее общая экспозиция)")


def parse_points(s: str) -> list[tuple[int, int]]:
    points = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_str, y_str = chunk.split(",")
        points.append((int(x_str), int(y_str)))
    return points


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def process_one(in_path: Path, radius_arg, strength: float, out_dir: Path,
                 save_maps: bool, sample_points, mode: str, threshold: float,
                 max_correction, power: float, **extra):
    with Image.open(in_path) as im:
        w, h = im.size
    radius = radius_arg if radius_arg is not None else max(w, h) * 0.08

    out_path = out_dir / f"{in_path.stem}_flattened{in_path.suffix}"
    map_path = out_dir / f"{in_path.stem}_correction_map{in_path.suffix}"
    tmp_map_path = out_dir / f".__tmp_map_{in_path.stem}{in_path.suffix}"

    flatten_exposure(
        str(in_path), radius, strength, str(out_path),
        str(map_path) if save_maps else str(tmp_map_path),
        sample_points, mode=mode, threshold=threshold,
        max_correction=max_correction, power=power, **extra,
    )

    if not save_maps and tmp_map_path.exists():
        tmp_map_path.unlink()


def process_batch(in_dir: Path, radius_arg, strength: float, out_dir: Path,
                   save_maps: bool, recursive: bool, mode: str, threshold: float,
                   max_correction, power: float, **extra):
    out_dir_resolved = out_dir.resolve()
    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in in_dir.glob(pattern)
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTS
        and out_dir_resolved not in f.resolve().parents  # не трогаем уже готовые результаты
        and "_flattened" not in f.stem
        and "_correction_map" not in f.stem
    )

    if not files:
        print(f"В папке {in_dir} не найдено изображений ({', '.join(IMAGE_EXTS)})", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Найдено файлов: {len(files)}. Результаты будут сохранены в: {out_dir}\n")

    ok, failed = 0, []
    for i, f in enumerate(files, 1):
        print(f"===== [{i}/{len(files)}] {f.name} =====")
        try:
            process_one(f, radius_arg, strength, out_dir, save_maps, sample_points=None,
                        mode=mode, threshold=threshold, max_correction=max_correction,
                        power=power, **extra)
            ok += 1
        except Exception as e:
            print(f"      ОШИБКА при обработке {f.name}: {e}", file=sys.stderr)
            failed.append(f.name)
        print()

    print(f"Готово: {ok}/{len(files)} успешно.")
    if failed:
        print(f"Не удалось обработать: {', '.join(failed)}")


def main():
    p = argparse.ArgumentParser(description="Математическое выравнивание экспозиции (flat-field correction)")
    p.add_argument("input", help="путь к изображению ИЛИ к папке с изображениями (для пакетной обработки)")
    p.add_argument("--radius", type=float, default=None,
                    help="радиус Gaussian blur в пикселях (по умолчанию 8%% от короткой стороны каждого фото)")
    p.add_argument("--strength", type=float, default=0.5,
                    help="сила коррекции 0..1 (рекомендуется 0.3-0.6, по умолчанию 0.5)")
    p.add_argument("--out", default=None,
                    help="файл результата (для одиночного фото) или папка результатов (для пакетной обработки)")
    p.add_argument("--map", default=None, help="путь для карты поправки (только одиночный режим)")
    p.add_argument("--points", default=None,
                    help="точки для проверки, формат 'x1,y1;x2,y2' (только одиночный режим)")
    p.add_argument("--no-maps", action="store_true",
                    help="не сохранять диагностические карты поправки (пакетный режим, экономит место/время)")
    p.add_argument("--recursive", action="store_true",
                    help="обходить папку рекурсивно, включая подпапки (пакетный режим)")
    p.add_argument("--mode", choices=["both", "darks", "lights"], default="both",
                    help="'both' — правит и тени, и света (по умолчанию); "
                         "'darks' — трогает только зоны темнее среднего (осветляет тени); "
                         "'lights' — трогает только зоны светлее среднего (притемняет пересветы)")
    p.add_argument("--threshold", type=float, default=0.0,
                    help="чувствительность/мёртвая зона в пунктах L (0-100). "
                         "Зоны с отклонением от среднего меньше этого значения не трогаются. "
                         "0 = правится всё, даже мелкие расхождения. 5-10 = трогаются только заметные перепады")
    p.add_argument("--max-correction", type=float, default=None,
                    help="потолок силы поправки в пунктах L (после strength). "
                         "Защищает глубокие тени/яркие света от переисправления. Например 15")
    p.add_argument("--power", type=float, default=1.0,
                    help="кривая чувствительности к величине отклонения. "
                         "1.0 = линейно (по умолчанию); <1 (напр. 0.6) = агрессивнее на мелких перепадах; "
                         ">1 (напр. 1.5) = мягче на мелких, сильнее только на явных перепадах")

    # --- умное определение target ---
    p.add_argument("--target-mode", choices=["mean", "median", "trimmed", "mode"], default="trimmed",
                    help="как определять 'мишень' (target). 'trimmed' (по умолчанию) — обрезанное "
                         "среднее, устойчиво к выбросам (блики, рамки). 'median' — медиана. "
                         "'mode' — пик гистограммы (доминирующий тон). 'mean' — наивное среднее (старое поведение)")
    p.add_argument("--trim-percent", type=float, default=5.0,
                    help="для --target-mode trimmed: сколько процентов самых тёмных и самых светлых "
                         "пикселей отбросить перед усреднением (по умолчанию 5)")
    p.add_argument("--target", type=float, default=None,
                    help="ручное значение target (0-100), игнорирует автоматику. "
                         "Используйте, когда автоматика не понимает художественный замысел кадра")
    p.add_argument("--no-adaptive", action="store_true",
                    help="выключить автоматическое ослабление strength при экстремальном target "
                         "(по умолчанию адаптация включена — защищает low-key/high-key сцены от пере-коррекции)")
    p.add_argument("--clip-knee", type=float, default=6.0,
                    help="мягкость ограничения диапазона 0-100 в пунктах L (0 = жёсткий clip). "
                         "По умолчанию 6 — небольшое смягчение без явного 'плёночного' эффекта")

    # --- плёночный look ---
    p.add_argument("--film", action="store_true",
                    help="включить плёночный тон-курв поверх коррекции экспозиции")
    p.add_argument("--film-lift", type=float, default=4.0,
                    help="подъём чёрной точки для --film, пункты L (по умолчанию 4)")
    p.add_argument("--film-rolloff", type=float, default=12.0,
                    help="мягкость сжатия светов для --film, пункты L (по умолчанию 12)")
    p.add_argument("--film-warmth", type=float, default=0.3,
                    help="сила split-toning для --film: тёплые тени/холодные света, 0-1 (по умолчанию 0.3)")
    p.add_argument("--film-grain", type=float, default=0.15,
                    help="сила зерна для --film, 0-1 (по умолчанию 0.15, 0 = без зерна)")
    p.add_argument("--seed", type=int, default=0, help="seed для генератора зерна (--film-grain), для воспроизводимости")

    # --- ROI: отделение объекта от рамы/паспарту/фона ---
    p.add_argument("--roi", default=None,
                    help="ручная область объекта в пикселях 'x,y,ширина,высота'. "
                         "Target и коррекция считаются ТОЛЬКО внутри — рама/паспарту игнорируются")
    p.add_argument("--roi-percent", type=float, default=None,
                    help="ручная область объекта как %% от центра кадра (напр. 80 = центральные 80%% "
                         "по ширине и высоте, отступ по 10%% с каждого края). Проще чем --roi, "
                         "хорошо работает при единообразной композиции в пакете")
    p.add_argument("--auto-roi", action="store_true",
                    help="автоматически определить объект по контрасту с рамкой/окружением "
                         "(сравнивает цвет тонкой полосы по периметру кадра с остальным изображением). "
                         "Можно сочетать с --roi/--roi-percent для доп. защиты")
    p.add_argument("--roi-sensitivity", type=float, default=4.0,
                    help="для --auto-roi: чувствительность отделения объекта от фона в стандартных "
                         "отклонениях (по умолчанию 4.0). Меньше = маска объекта шире/смелее, "
                         "больше = маска строже/уже")
    p.add_argument("--roi-feather", type=float, default=None,
                    help="растушёвка границы ROI в пикселях, чтобы не было резкого шва "
                         "между скорректированным объектом и нетронутой рамой (по умолчанию ~1.5%% "
                         "от короткой стороны кадра)")
    p.add_argument("--correction-scope", choices=["roi", "full"], default="roi",
                    help="'roi' (по умолчанию) — target и коррекция ограничены объектом, рама не трогается. "
                         "'full' — target берётся только из объекта (ROI), но коррекция применяется "
                         "ко ВСЕМУ кадру, включая раму/паспарту/фон")

    # --- Dodge & Burn: усиление объёма ---
    p.add_argument("--dodge-burn", action="store_true",
                    help="включить автоматический Dodge & Burn — усиливает объём/пластику "
                         "(противоположность выравниванию экспозиции: не гасит локальные "
                         "перепады, а усиливает их)")
    p.add_argument("--db-radius", type=float, default=60.0,
                    help="масштаб объёма для Dodge & Burn, пикселей (по умолчанию 60). "
                         "Маленький (5-15) = микро-контраст/текстура; средний (30-80) = объём "
                         "форм и складок; большой (100+) = крупная лепка всей композиции")
    p.add_argument("--db-strength", type=float, default=0.3,
                    help="сила усиления объёма, 0-1+ (по умолчанию 0.3). 0.15-0.25 деликатно, "
                         "0.4-0.6 заметно, 0.7+ риск пересвеченных/пережжённых краёв (halo)")
    p.add_argument("--db-max", type=float, default=15.0,
                    help="потолок усиления в пунктах L, защита от halo на резких границах "
                         "(по умолчанию 15, None/0 = без ограничения)")

    # --- Halation: плёночное свечение бликов ---
    p.add_argument("--halation", action="store_true",
                    help="включить halation — тёплое свечение вокруг ярких участков/бликов, "
                         "характерное для цветной плёнки")
    p.add_argument("--halation-threshold", type=float, default=80.0,
                    help="с какой яркости L (0-100) начинается свечение (по умолчанию 80 — "
                         "только настоящие света/блики, не средние тона)")
    p.add_argument("--halation-strength", type=float, default=0.3,
                    help="сила свечения, 0-1 (по умолчанию 0.3)")
    p.add_argument("--halation-radius", type=float, default=15.0,
                    help="радиус растекания свечения в пикселях (по умолчанию 15)")

    # --- Виньетирование ---
    p.add_argument("--vignette", action="store_true",
                    help="включить плавное затемнение углов кадра (оптическая характеристика "
                         "плёночных/винтажных объективов)")
    p.add_argument("--vignette-strength", type=float, default=0.25,
                    help="сила затемнения углов, 0-1 (по умолчанию 0.25)")
    p.add_argument("--vignette-radius", type=float, default=0.85,
                    help="на каком относительном расстоянии от центра (0-1) начинается "
                         "затемнение (по умолчанию 0.85)")
    p.add_argument("--vignette-softness", type=float, default=0.5,
                    help="мягкость перехода виньетки, 0-1 (по умолчанию 0.5, больше = плавнее)")

    args = p.parse_args()
    in_path = Path(args.input)

    if not in_path.exists():
        print(f"Путь не найден: {in_path}", file=sys.stderr)
        sys.exit(1)

    roi_rect = None
    if args.roi:
        parts = [int(v) for v in args.roi.split(",")]
        if len(parts) != 4:
            print("--roi должен быть в формате 'x,y,ширина,высота'", file=sys.stderr)
            sys.exit(1)
        roi_rect = tuple(parts)

    extra = dict(
        target_mode=args.target_mode, trim_percent=args.trim_percent,
        manual_target=args.target, adaptive=not args.no_adaptive, clip_knee=args.clip_knee,
        film=args.film, film_lift=args.film_lift, film_rolloff=args.film_rolloff,
        film_warmth=args.film_warmth, film_grain=args.film_grain, seed=args.seed,
        roi_rect=roi_rect, roi_percent=args.roi_percent, auto_roi=args.auto_roi,
        roi_sensitivity=args.roi_sensitivity, roi_feather=args.roi_feather,
        correction_scope=args.correction_scope,
        dodge_burn=args.dodge_burn, db_radius=args.db_radius,
        db_strength=args.db_strength, db_max=args.db_max,
        halation=args.halation, halation_threshold=args.halation_threshold,
        halation_strength=args.halation_strength, halation_radius=args.halation_radius,
        vignette=args.vignette, vignette_strength=args.vignette_strength,
        vignette_radius=args.vignette_radius, vignette_softness=args.vignette_softness,
    )

    if in_path.is_dir():
        # ---- ПАКЕТНЫЙ РЕЖИМ ----
        out_dir = Path(args.out) if args.out else in_path / "flattened"
        process_batch(
            in_path, args.radius, args.strength, out_dir,
            save_maps=not args.no_maps, recursive=args.recursive,
            mode=args.mode, threshold=args.threshold,
            max_correction=args.max_correction, power=args.power, **extra,
        )
    else:
        # ---- ОДИНОЧНЫЙ РЕЖИМ ----
        with Image.open(in_path) as im:
            w, h = im.size
        radius = args.radius if args.radius is not None else max(w, h) * 0.08

        out_path = args.out or str(in_path.with_stem(in_path.stem + "_flattened"))
        map_path = args.map or str(in_path.with_stem(in_path.stem + "_correction_map"))

        sample_points = parse_points(args.points) if args.points else None

        flatten_exposure(str(in_path), radius, args.strength, out_path, map_path, sample_points,
                          mode=args.mode, threshold=args.threshold,
                          max_correction=args.max_correction, power=args.power, **extra)


if __name__ == "__main__":
    main()