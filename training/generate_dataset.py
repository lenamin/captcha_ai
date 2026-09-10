"""
합성 숫자 캡차 데이터 생성 (2 스타일).

- style A (warp): `captcha` 라이브러리. 문자 왜곡·노이즈가 강함.
- style B (strike): PIL로 6자리 숫자를 나란히 렌더 + 가운데 가로 취소선.

각 스타일별로:
  1) single/<digit>/*.png  — 학습용 단일 숫자 크롭(라벨 100%)
  2) multi/*.png           — 6자리 캡차(파일명 = 정답) end-to-end 평가용

Style B는 문자 위치를 우리가 직접 정하므로, multi 이미지를 만들면서 동시에
정확한 개별 숫자 크롭을 저장해 라벨 노이즈를 없앴다.
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from captcha.image import ImageCaptcha

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "synthetic_dataset"

STYLE_A_SINGLE = DATA / "styleA" / "single"
STYLE_A_MULTI = DATA / "styleA" / "multi"
STYLE_B_SINGLE = DATA / "styleB" / "single"
STYLE_B_MULTI = DATA / "styleB" / "multi"
STYLE_B_MULTI_AUG = DATA / "styleB" / "multi_aug"

# 가용한 truetype 폰트 리스트 (일반적으로 데비안/우분투에 설치됨)
CANDIDATE_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
]
FONTS = [f for f in CANDIDATE_FONTS if Path(f).exists()]


# --------------------------------------------------------------------- style A
def gen_style_a(per_class: int, multi_count: int, seed: int) -> None:
    rng = random.Random(seed)
    single_eng = ImageCaptcha(width=60, height=60, font_sizes=(42, 46, 50))
    multi_eng = ImageCaptcha(width=200, height=70)

    STYLE_A_SINGLE.mkdir(parents=True, exist_ok=True)
    for d in string.digits:
        (STYLE_A_SINGLE / d).mkdir(exist_ok=True)
        for i in range(per_class):
            single_eng.write(d, str(STYLE_A_SINGLE / d / f"{i:05d}.png"))
    print(f"  styleA single: {per_class*10} imgs")

    STYLE_A_MULTI.mkdir(parents=True, exist_ok=True)
    for i in range(multi_count):
        s = "".join(rng.choice(string.digits) for _ in range(6))
        multi_eng.write(s, str(STYLE_A_MULTI / f"{i:04d}_{s}.png"))
    print(f"  styleA multi (6-digit): {multi_count} imgs")


# --------------------------------------------------------------------- style B
def _render_strike_multi(
    digits: str,
    rng: random.Random,
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """6자리 취소선 스타일 캡차 렌더링. (이미지, 각 자릿수 bbox 리스트) 반환.

    다양성을 위해 다음 파라미터를 랜덤화한다:
      - 폰트 종류·크기
      - 글자 회전각(±8°)·y 지터
      - 수평 취소선 개수(0~2)·두께·y 위치
      - 대각선 noise line (30% 확률)
      - 짧은 랜덤 선분 3~8개
      - salt-and-pepper 노이즈 밀도(0.002~0.02)
      - 배경 gray dot 노이즈
    """
    W, H = 220, 70
    im = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(im)

    # 문자 회전을 지원하기 위해 개별 문자를 별도 캔버스에 렌더 후 붙임
    font_path = rng.choice(FONTS) if FONTS else None
    size = rng.randint(34, 52)
    font = (ImageFont.truetype(font_path, size)
            if font_path else ImageFont.load_default(size))

    # 각 자릿수 실제 렌더 폭 계산
    widths, heights = [], []
    for ch in digits:
        bbox = draw.textbbox((0, 0), ch, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    total_w = sum(widths)
    # 최소 간격 5px 보장 (회전·지터 후에도 겹치지 않도록)
    gap = max(5, (W - total_w) // (len(digits) + 1))
    x = gap
    boxes = []
    for ch, w, h in zip(digits, widths, heights):
        # 개별 문자를 흰 배경 캔버스에 그린 뒤 회전 → 마스크 합성
        pad = 6
        cw, chgt = w + pad * 2, h + pad * 2
        ch_img = Image.new("L", (cw, chgt), 255)
        ch_draw = ImageDraw.Draw(ch_img)
        ch_draw.text((pad, pad), ch, font=font, fill=0)
        angle = rng.uniform(-8, 8)
        ch_img = ch_img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        # 회전으로 확장된 bbox의 실제 잉크 영역만 사용
        arr = np.asarray(ch_img)
        ys, xs = np.where(arr < 128)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tight = ch_img.crop((x0, y0, x1, y1))
        tw, th = tight.size

        y = (H - th) // 2 + rng.randint(-4, 4)
        # 흰 배경 이미지에 검정만 페이스트
        mask = tight.point(lambda v: 255 if v < 128 else 0)
        im.paste(0, (x, y, x + tw, y + th), mask=mask)
        boxes.append((x, y, x + tw, y + th))
        # 지터도 최소 간격을 깨지 않게 클램프
        x += tw + max(5, gap + rng.randint(-2, 3))

    # 가로 취소선 0~2개
    n_lines = rng.choices([0, 1, 2], weights=[1, 6, 3])[0]
    for _ in range(n_lines):
        y_line = rng.randint(H // 3, 2 * H // 3)
        thick = rng.randint(1, 4)
        x_start = rng.randint(0, 10)
        x_end = W - rng.randint(0, 10)
        draw.line([(x_start, y_line), (x_end, y_line)], fill=0, width=thick)

    # 대각선 noise line (30%)
    if rng.random() < 0.3:
        y1 = rng.randint(5, H - 5)
        y2 = rng.randint(5, H - 5)
        draw.line([(0, y1), (W, y2)], fill=rng.randint(0, 60),
                  width=rng.randint(1, 2))

    # 짧은 랜덤 선분 3~8개
    for _ in range(rng.randint(3, 8)):
        x1 = rng.randint(0, W)
        y1 = rng.randint(0, H)
        x2 = x1 + rng.randint(-25, 25)
        y2 = y1 + rng.randint(-15, 15)
        draw.line([(x1, y1), (x2, y2)],
                  fill=rng.randint(0, 90), width=1)

    # salt-and-pepper 노이즈
    arr = np.asarray(im).copy()
    density = rng.uniform(0.002, 0.02)
    smask = np.random.rand(H, W) < density
    arr[smask] = 0
    # 배경 gray dot 노이즈
    gmask = np.random.rand(H, W) < 0.01
    arr[gmask] = np.where(arr[gmask] > 128,
                          np.random.randint(120, 200, size=gmask.sum()),
                          arr[gmask])
    return Image.fromarray(arr), boxes


def gen_style_b(per_class: int, multi_count: int, seed: int) -> None:
    """style B: 두 벌의 multi를 만든다 — eval 전용(multi/) + train-augmentation(multi_aug/).
    single/ 는 known-bbox 크롭(정답 정확)으로 채운다.
    per_class는 목표치 — 실제로는 multi 렌더 수에 따라 자동으로 채워짐."""
    rng = random.Random(seed)
    np.random.seed(seed)

    STYLE_B_SINGLE.mkdir(parents=True, exist_ok=True)
    for d in string.digits:
        (STYLE_B_SINGLE / d).mkdir(exist_ok=True)
    STYLE_B_MULTI.mkdir(parents=True, exist_ok=True)
    STYLE_B_MULTI_AUG.mkdir(parents=True, exist_ok=True)

    # 목표: (1) single per_class 만큼, (2) eval multi_count 개, (3) aug multi_count 개
    counters = {d: 0 for d in string.digits}
    n_eval = 0
    n_aug = 0
    idx = 0
    aug_target = multi_count  # 학습보강용 multi도 동수만큼 생성
    while (min(counters.values()) < per_class
           or n_eval < multi_count
           or n_aug < aug_target):
        s = "".join(rng.choice(string.digits) for _ in range(6))
        im, boxes = _render_strike_multi(s, rng)

        # eval 우선 채우고, 남으면 aug로 저장
        if n_eval < multi_count:
            im.save(STYLE_B_MULTI / f"{n_eval:04d}_{s}.png")
            n_eval += 1
        elif n_aug < aug_target:
            im.save(STYLE_B_MULTI_AUG / f"{n_aug:04d}_{s}.png")
            n_aug += 1

        # 개별 자릿수 크롭(known bbox) 저장
        for ch, (x0, y0, x1, y1) in zip(s, boxes):
            if counters[ch] >= per_class:
                continue
            pad = 3
            crop = im.crop((max(0, x0 - pad), max(0, y0 - pad),
                            min(im.size[0], x1 + pad),
                            min(im.size[1], y1 + pad)))
            crop.save(STYLE_B_SINGLE / ch / f"{counters[ch]:05d}.png")
            counters[ch] += 1
        idx += 1

    total = sum(counters.values())
    print(f"  styleB single(known-bbox): {total} imgs "
          f"(min/class={min(counters.values())})")
    print(f"  styleB multi eval: {n_eval} imgs, multi_aug (학습보강): {n_aug} imgs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--multi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[A] captcha 라이브러리 스타일 생성 …")
    gen_style_a(args.per_class, args.multi, args.seed)
    print("[B] 취소선 스타일 생성 …")
    gen_style_b(args.per_class, args.multi, args.seed + 1)
    print("완료.")


if __name__ == "__main__":
    main()
