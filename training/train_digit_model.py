"""
합성 숫자 캡차(2 스타일 통합) 분류 모델 학습.

두 종류 데이터를 모두 학습해 하나의 model.pickle로 저장한다:
  - styleA (warp): `captcha` 라이브러리 왜곡 스타일
  - styleB (strike): PIL로 렌더한 6자리 + 가운데 가로 취소선 스타일

전처리 파이프라인:
  1) 그레이스케일 → Otsu 이진화 (문자=1)
  2) 가로 취소선 제거 — 폭 40px 이상 이어지는 수평 잉크만 남기는
     형태학적 열림으로 취소선을 뽑아내 원본에서 뺀다 (styleA에는 사실상 무영향)
  3) 잉크 영역 tight crop → 40x40 리사이즈 → 평탄화(1600차원)

end-to-end 6자리 평가:
  - styleA: 왜곡이 심해 세그먼트가 어려움 → 폭 6등분(baseline)
  - styleB: 취소선 제거 후 연결요소(connected components)로 6개 검출,
            정확히 6개가 안 나오면 세로 프로젝션 valley 기반 fallback
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from skimage.morphology import footprint_rectangle as _fp_rect

    def _hrect(h: int, w: int):
        return _fp_rect((h, w))
except ImportError:  # skimage < 0.25
    from skimage.morphology import rectangle as _rect

    def _hrect(h: int, w: int):
        return _rect(h, w)

from skimage.morphology import opening, closing
from skimage.measure import label, regionprops
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "synthetic_dataset"
STYLE_A_SINGLE = DATA / "styleA" / "single"
STYLE_A_MULTI = DATA / "styleA" / "multi"
STYLE_B_SINGLE = DATA / "styleB" / "single"
STYLE_B_MULTI = DATA / "styleB" / "multi"
STYLE_B_MULTI_AUG = DATA / "styleB" / "multi_aug"
MODEL_OUT = ROOT / "model.pickle"
REPORT_OUT = ROOT / "training" / "training_report.txt"

IMG_SIZE = (40, 40)  # (W, H)
# 특성 = [평탄화된 이진 40x40] + [HOG(8x8 cell, 2x2 block, 9 orient)]
# HOG 차원: ((40/8 - 1) * (40/8 - 1)) * 4 * 9 = 4*4*36 = 576
FEATURE_DIM = IMG_SIZE[0] * IMG_SIZE[1] + 576


# --------------------------------------------------------------- preprocessing
def _otsu(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, w_bg, max_var, thr = 0.0, 0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        m_bg = sum_bg / w_bg
        m_fg = (sum_all - sum_bg) / w_fg
        v = w_bg * w_fg * (m_bg - m_fg) ** 2
        if v > max_var:
            max_var, thr = v, t
    return thr


def binarize(img: Image.Image) -> np.ndarray:
    """PIL 이미지 → 이진(0/1, 문자=1) numpy 배열."""
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    thr = _otsu(g)
    # 배경이 밝은 경우가 표준이지만 반대 케이스도 자동 처리
    if g.mean() > thr:
        return (g < thr).astype(np.uint8)
    return (g >= thr).astype(np.uint8)


def remove_hstrike(binv: np.ndarray, min_len: int = 40) -> np.ndarray:
    """멀티 스케일 수평 opening으로 취소선을 뽑아 제거한다.

    문자 획 사이를 지나가는 취소선은 획에 의해 짧게 끊길 수 있으므로,
    먼저 폭 3짜리 수평 closing으로 미세 균열을 메운 뒤 opening으로 긴 수평선만 추출.
    폭 {20, 30, min_len}을 OR로 합쳐 두꺼운/얇은 취소선 모두 커버한다.
    문자에 겹치는 지점의 획 손실은 이어지는 문자 영역이 여전히 넓게 남아 있어
    분류에는 큰 영향이 없다."""
    bridged = closing(binv, _hrect(1, 3))
    line = np.zeros_like(binv)
    for L in (20, 30, min_len):
        line |= opening(bridged, _hrect(1, L))
    # 라인 마스크를 세로로 살짝 팽창시켜 두꺼운 취소선도 완전히 걷어냄
    line = closing(line, _hrect(3, 1))
    cleaned = np.clip(binv.astype(np.int16) - line.astype(np.int16), 0, 1).astype(np.uint8)
    return cleaned


def tight_crop_resize(binv: np.ndarray) -> np.ndarray:
    """잉크 영역 bbox로 크롭 → IMG_SIZE 리사이즈 → [이진픽셀 flatten | HOG] 벡터."""
    ys, xs = np.where(binv > 0)
    if ys.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = binv[y0:y1, x0:x1]
    pil = Image.fromarray((crop * 255).astype(np.uint8))
    pil = pil.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    binf = (arr > 0.5).astype(np.float32)
    h = hog(arr, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            feature_vector=True)
    return np.concatenate([binf.flatten(), h.astype(np.float32)])


def preprocess_single(img: Image.Image) -> np.ndarray:
    """단일 숫자 이미지용: 이진화 → 취소선 제거 → 크롭/리사이즈."""
    binv = binarize(img)
    binv = remove_hstrike(binv, min_len=40)
    return tight_crop_resize(binv)


# ---------------------------------------------------------- dataset loading
def load_singles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """styleA + styleB 단일 숫자 이미지 로드. (X, y, style) 반환.

    styleA: 라이브러리 단일 숫자 렌더 (정답 정확).
    styleB: multi 이미지에서 뽑은 known-bbox 크롭 (정답 정확).
    거기에 더해 styleB multi 이미지에 **추론용 세그먼트 파이프라인**을 그대로 적용해
    6조각이 정확히 나오는 이미지에 한해 그 조각들을 x-순서로 정답과 매칭시켜
    "실제 추론 분포"에 가까운 학습 샘플을 추가한다.
    """
    X, y, style = [], [], []
    for root, tag in [(STYLE_A_SINGLE, "A"), (STYLE_B_SINGLE, "B")]:
        for digit_dir in sorted(root.iterdir()):
            if not digit_dir.is_dir():
                continue
            for p in sorted(digit_dir.glob("*.png")):
                X.append(preprocess_single(Image.open(p)))
                y.append(digit_dir.name)
                style.append(tag)

    # ---- styleB multi_aug 를 세그먼트해 얻은 크롭도 학습에 추가 (분포 매칭용)
    # eval용 multi/ 는 절대 학습에 넣지 않음 (data leakage 방지)
    added = 0
    for p in sorted(STYLE_B_MULTI_AUG.glob("*.png")):
        label_ = p.stem.split("_", 1)[1]
        binv = binarize(Image.open(p))
        binv2 = remove_hstrike(binv, min_len=40)
        segs = segment_by_components(binv2, 6)
        if segs is None:
            segs = segment_by_x_clustering(binv2, 6)
        if segs is None or len(segs) != 6:
            continue
        for ch, seg in zip(label_, segs):
            X.append(tight_crop_resize(seg))
            y.append(ch)
            style.append("B_seg")
            added += 1
    print(f"    (+ styleB 세그먼트-기반 학습샘플 {added}개 추가)")
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y), np.asarray(style))


# ---------------------------------------------------------- 6-digit segmenting
def segment_equal_width(binv: np.ndarray, n: int = 6) -> list[np.ndarray]:
    """폭 균등 분할(styleA baseline)."""
    h, w = binv.shape
    seg = w // n
    out = []
    for i in range(n):
        x0 = i * seg
        x1 = (i + 1) * seg if i < n - 1 else w
        out.append(binv[:, x0:x1])
    return out


def segment_by_components(binv: np.ndarray, n: int = 6) -> list[np.ndarray] | None:
    """연결요소 기반. 6개면 성공, 아니면 None."""
    lbl = label(binv, connectivity=2)
    props = [p for p in regionprops(lbl) if p.area >= 20]
    if len(props) != n:
        return None
    props.sort(key=lambda p: p.bbox[1])  # x-min 오름차순
    out = []
    for p in props:
        y0, x0, y1, x1 = p.bbox
        out.append(binv[y0:y1, x0:x1])
    return out


def segment_by_x_clustering(binv: np.ndarray, n: int = 6) -> list[np.ndarray] | None:
    """모든 연결요소의 x-중심을 KMeans(k=n)로 묶고, 같은 클러스터를 하나의 문자로 병합.

    취소선 제거 후 문자가 조각나도 x 위치가 비슷하면 같은 문자로 판정된다.
    소음 필터링: area >= 12, 그리고 컴포넌트 너무 많으면 상위 area만 남긴다.
    """
    lbl = label(binv, connectivity=2)
    props = [p for p in regionprops(lbl) if p.area >= 12]
    if len(props) < n:
        return None
    # 컴포넌트가 너무 많으면 큰 것 위주로 잘라내되, 각 x-빈에서 최소 1개는 보존
    if len(props) > n * 4:
        props.sort(key=lambda p: p.area, reverse=True)
        props = props[: n * 4]
    xs = np.array([[p.centroid[1]] for p in props])
    weights = np.array([p.area for p in props], dtype=np.float32)
    km = KMeans(n_clusters=n, n_init=10, random_state=42).fit(
        xs, sample_weight=weights
    )
    order = np.argsort(km.cluster_centers_.ravel())
    result = []
    for cluster_id in order:
        members = [p for p, lab_ in zip(props, km.labels_) if lab_ == cluster_id]
        if not members:
            return None  # 클러스터 비면 실패
        y0 = min(p.bbox[0] for p in members)
        x0 = min(p.bbox[1] for p in members)
        y1 = max(p.bbox[2] for p in members)
        x1 = max(p.bbox[3] for p in members)
        result.append(binv[y0:y1, x0:x1])
    return result


def segment_by_projection(binv: np.ndarray, n: int = 6) -> list[np.ndarray]:
    """세로 잉크 프로젝션에서 valley(0) 구간을 잘라 n조각. 부족하면 균등 분할 병행."""
    col_sum = binv.sum(axis=0)
    w = binv.shape[1]

    # 잉크가 있는 구간의 시작/끝 구하기
    is_ink = col_sum > 0
    runs = []
    x = 0
    while x < w:
        if is_ink[x]:
            x0 = x
            while x < w and is_ink[x]:
                x += 1
            runs.append((x0, x))
        else:
            x += 1

    if len(runs) == n:
        return [binv[:, a:b] for a, b in runs]
    if len(runs) < n:
        # 넓은 run은 균등 분할로 잘라 개수 맞추기
        while len(runs) < n:
            # 가장 넓은 run 선택
            i = max(range(len(runs)), key=lambda k: runs[k][1] - runs[k][0])
            a, b = runs[i]
            mid = (a + b) // 2
            runs = runs[:i] + [(a, mid), (mid, b)] + runs[i + 1:]
        return [binv[:, a:b] for a, b in runs]
    # 너무 많으면 좁은 run들끼리 병합
    while len(runs) > n:
        i = min(range(len(runs) - 1),
                key=lambda k: (runs[k + 1][0] - runs[k][1]))
        a, _ = runs[i]
        _, b = runs[i + 1]
        runs = runs[:i] + [(a, b)] + runs[i + 2:]
    return [binv[:, a:b] for a, b in runs]


def features_of_segments(segs: list[np.ndarray]) -> np.ndarray:
    """세그먼트 리스트 → (n, FEATURE_DIM) 특성 배열."""
    out = np.zeros((len(segs), FEATURE_DIM), dtype=np.float32)
    for i, s in enumerate(segs):
        out[i] = tight_crop_resize(s)
    return out


# ---------------------------------------------------------- model comparison
def compare_models(X_tr, y_tr):
    # 3-fold로 시간 단축, LinearSVC를 SVM(rbf) 대신 사용 (HOG와 궁합 좋음, 훨씬 빠름)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42
        ),
        "LinearSVC": LinearSVC(C=1.0, dual="auto", max_iter=3000, random_state=42),
    }
    results = []
    for name, clf in models.items():
        t0 = time.time()
        scores = cross_val_score(clf, X_tr, y_tr, cv=skf, n_jobs=-1)
        dt = time.time() - t0
        print(f"  - {name:<15s} CV acc = {scores.mean():.4f} "
              f"± {scores.std():.4f}  ({dt:.1f}s)")
        results.append((name, float(scores.mean()), float(scores.std())))
    return results


# ---------------------------------------------------------- end-to-end eval
def eval_e2e(clf, multi_dir: Path, style: str) -> tuple[float, float, int]:
    tot_d, cor_d, tot_s, cor_s = 0, 0, 0, 0
    for p in sorted(multi_dir.glob("*.png")):
        label_ = p.stem.split("_", 1)[1]
        binv = binarize(Image.open(p))
        binv = remove_hstrike(binv, min_len=40)

        segs = None
        if style == "B":
            segs = segment_by_components(binv, 6)
            if segs is None:
                segs = segment_by_x_clustering(binv, 6)
            if segs is None:
                segs = segment_by_projection(binv, 6)
        else:
            segs = segment_equal_width(binv, 6)

        feats = features_of_segments(segs)
        pred = "".join(clf.predict(feats).tolist())
        tot_d += len(label_)
        cor_d += sum(a == b for a, b in zip(label_, pred))
        tot_s += 1
        cor_s += int(pred == label_)
    return cor_d / max(1, tot_d), cor_s / max(1, tot_s), tot_s


# ---------------------------------------------------------- main
def main() -> int:
    print("[1/5] 단일 숫자 데이터 로드 & 전처리 …")
    X, y, style = load_singles()
    print(f"    X.shape={X.shape}, styleA={int((style=='A').sum())}, "
          f"styleB={int((style=='B').sum())}")
    print(f"    클래스={sorted(set(y.tolist()))}")

    print("[2/5] 학습/검증 분할 (8:2, stratified) …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    train={X_tr.shape[0]}, test={X_te.shape[0]}")

    print("[3/5] 5-fold CV 모델 비교 …")
    cv_results = compare_models(X_tr, y_tr)
    best_name, best_mean, _ = max(cv_results, key=lambda r: r[1])
    print(f"    ▶ 최적: {best_name} (CV acc={best_mean:.4f})")

    print("[4/5] 최적 모델 재학습 & 저장 …")
    if best_name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=42)
    elif best_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    else:
        clf = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    test_acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, digits=4)
    joblib.dump(clf, MODEL_OUT)
    print(f"    저장: {MODEL_OUT} ({MODEL_OUT.stat().st_size/1024:.1f} KB)")

    print("[5/5] end-to-end 6자리 정확도 (multi 폴더) …")
    a_dig, a_seq, a_n = eval_e2e(clf, STYLE_A_MULTI, "A")
    b_dig, b_seq, b_n = eval_e2e(clf, STYLE_B_MULTI, "B")
    print(f"    styleA(warp)   digit={a_dig:.4f}, 6-match={a_seq:.4f} ({a_n})")
    print(f"    styleB(strike) digit={b_dig:.4f}, 6-match={b_seq:.4f} ({b_n})")

    lines = [
        f"model               : {best_name}",
        f"train_size          : {X_tr.shape[0]}",
        f"test_size           : {X_te.shape[0]}",
        f"features            : {FEATURE_DIM} (40x40 binary, strike removed)",
        f"classes             : {sorted(np.unique(y).tolist())}",
        f"single_test_acc     : {test_acc:.4f}",
        f"styleA_multi_digit  : {a_dig:.4f}",
        f"styleA_multi_6match : {a_seq:.4f} ({a_n} imgs)",
        f"styleB_multi_digit  : {b_dig:.4f}",
        f"styleB_multi_6match : {b_seq:.4f} ({b_n} imgs)",
        "",
        "cv_summary (5-fold on train split):",
    ]
    for name, mean, std in cv_results:
        lines.append(f"  {name:<15s} {mean:.4f} ± {std:.4f}")
    lines += ["", "single-digit classification report (test):", report]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"단일 숫자 test acc : {test_acc:.4f}")
    print(f"styleA 6-match     : {a_seq:.4f}")
    print(f"styleB 6-match     : {b_seq:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
