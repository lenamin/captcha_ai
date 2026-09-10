"""
CAPTCHA 글자 단위 분류 모델 학습 스크립트.

이 레포는 6글자 알파벳 캡차용으로,
`datasets/d2.csv`, `d3.csv`, `d4.csv`에 이미 글자 단위로 전처리된
라벨 데이터(1401 컬럼 = 라벨 1 + 픽셀 1400)가 들어 있다.

학습된 최적 모델을 `model.pickle`(joblib 직렬화)로 저장한다.

빅데이터분석기사 관점의 실습 흐름:
  1) 데이터 로딩·병합
  2) 학습/검증 분할 + StratifiedKFold 교차검증
  3) 여러 분류기 비교 (DecisionTree / RandomForest / SVM)
  4) 최적 모델 재학습 후 저장 + 리포트 출력
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
MODEL_OUT = ROOT / "model.pickle"
REPORT_OUT = ROOT / "training" / "training_report.txt"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """d2, d3, d4를 병합해 (X, y) 반환. d1은 열 수가 불규칙해 제외."""
    frames = []
    for name in ("d2.csv", "d3.csv", "d4.csv"):
        path = DATASETS / name
        df = pd.read_csv(path, header=None)
        # 컬럼 수 검증: 1(label) + 1400(features)
        if df.shape[1] != 1401:
            raise ValueError(f"{name} 컬럼 수 이상: {df.shape}")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    y = all_df.iloc[:, 0].astype(str).to_numpy()
    X = all_df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    return X, y


def compare_models(X_tr, y_tr) -> list[tuple[str, float, float]]:
    """3개 분류기를 5-fold CV로 비교. (이름, 평균정확도, 표준편차) 반환."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
        "SVM(rbf)": SVC(C=10, gamma=0.001, kernel="rbf", random_state=42),
    }
    results = []
    for name, clf in models.items():
        t0 = time.time()
        scores = cross_val_score(clf, X_tr, y_tr, cv=skf, n_jobs=-1)
        dt = time.time() - t0
        print(
            f"  - {name:<15s} CV acc = {scores.mean():.4f} "
            f"± {scores.std():.4f}  ({dt:.1f}s)"
        )
        results.append((name, float(scores.mean()), float(scores.std())))
    return results


def main() -> int:
    print("[1/4] 데이터 로딩 …")
    X, y = load_data()
    print(f"    총 샘플: {X.shape[0]}, 피처: {X.shape[1]}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"    클래스 수: {len(classes)} → {dict(zip(classes.tolist(), counts.tolist()))}")

    print("[2/4] 학습/검증 분할 (8:2, stratify) …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    train={X_tr.shape[0]}, test={X_te.shape[0]}")

    print("[3/4] 모델 비교 (5-fold CV, 학습셋 기준) …")
    cv_results = compare_models(X_tr, y_tr)
    best_name, best_mean, _ = max(cv_results, key=lambda r: r[1])
    print(f"    ▶ 최적: {best_name} (CV acc={best_mean:.4f})")

    print("[4/4] 최적 모델 재학습 → model.pickle 저장 …")
    if best_name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=42)
    elif best_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    else:
        clf = SVC(C=10, gamma=0.001, kernel="rbf", random_state=42)

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, digits=4)
    cm = confusion_matrix(y_te, y_pred, labels=sorted(np.unique(y)))

    joblib.dump(clf, MODEL_OUT)
    print(f"    저장: {MODEL_OUT} ({MODEL_OUT.stat().st_size/1024:.1f} KB)")

    lines = [
        f"model      : {best_name}",
        f"train_size : {X_tr.shape[0]}",
        f"test_size  : {X_te.shape[0]}",
        f"features   : {X.shape[1]} (35x40 flattened pixels, binary)",
        f"classes    : {sorted(np.unique(y).tolist())}",
        f"test_acc   : {acc:.4f}",
        "",
        "cv_summary (5-fold on train split):",
    ]
    for name, mean, std in cv_results:
        lines.append(f"  {name:<15s} {mean:.4f} ± {std:.4f}")
    lines += ["", "classification_report (test):", report]

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"    리포트: {REPORT_OUT}")

    print()
    print("=" * 60)
    print(f"테스트 정확도: {acc:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
