# CAPTCHA 숫자 인식 합성 학습 파이프라인

빅데이터분석기사 실습용으로, `captcha` 라이브러리 + PIL로 합성 숫자 캡차를 만들어
sklearn 분류기를 학습해 `model.pickle`로 저장하는 예제.

## 흐름

```
generate_dataset.py      →  synthetic_dataset/
   ├─ styleA (captcha 라이브러리 왜곡 스타일)
   │   ├─ single/<0-9>/*.png       # 학습용 단일 숫자
   │   └─ multi/*.png              # 6자리 end-to-end 평가용
   └─ styleB (PIL 취소선 스타일)
       ├─ single/<0-9>/*.png       # 학습용 단일 숫자 (known-bbox 크롭)
       ├─ multi/*.png              # eval 전용 6자리 캡차
       └─ multi_aug/*.png          # 학습 보강용 6자리 캡차 (세그먼트해서 사용)

train_digit_model.py     →  model.pickle + training/training_report.txt
```

## 실행

```bash
# 1) 데이터 생성 (per-class × 10클래스 = 스타일당 단일 숫자 수)
python3 training/generate_dataset.py --per-class 2000 --multi 800

# 2) 학습·평가·저장
python3 training/train_digit_model.py
```

## 파이프라인 상세

### 전처리
1. 그레이스케일 변환 → Otsu 이진화 (문자=1)
2. 취소선 제거 (multi-scale 수평 opening 합집합)
3. 잉크 bbox tight crop → 40×40 리사이즈
4. 특징 = [평탄화 이진 40×40 = 1600] + [HOG 576] = **2176차원**

### 세그먼트 (6자리 캡차 → 6개 자리로)
1. `segment_by_components`: 연결요소가 정확히 6개면 x-순서로 채택 (styleB 이상적 케이스)
2. `segment_by_x_clustering`: 연결요소를 x-축 KMeans(k=6) 클러스터링 → 조각난 취소선/노이즈에 강함 (styleB fallback)
3. `segment_by_projection`: 세로 잉크 프로젝션의 valley로 분할 (마지막 fallback)
4. styleA는 왜곡이 심해 폭 균등 분할이 사실상 유일한 baseline

### 모델
- DecisionTreeClassifier: baseline
- RandomForestClassifier(300 → 최종 500 trees)
- LinearSVC: HOG 특징에 궁합 좋음, 대용량에서 빠름

3-fold StratifiedKFold CV로 비교 → 최적 모델을 전체 학습셋에 재학습 후 `model.pickle`로 저장.

## 결과 지표

`training/training_report.txt`에 저장:
- 단일 숫자 test accuracy
- styleA/styleB 6자리 자릿수 정확도(각 자리 hit ratio)
- styleA/styleB 6자리 완전일치 정확도(6개 다 맞은 비율)
- 각 모델 CV 요약
- 클래스별 precision/recall/f1 리포트

## 주의사항

- **styleA(captcha 라이브러리)의 end-to-end**: 문자 간 심한 왜곡·중첩으로 규칙 기반 세그먼트가 근본적으로 어렵습니다. 실무에서는 CNN + CTC(시퀀스 인식) 같은 딥러닝 접근이 필요합니다. 이 파이프라인의 styleA end-to-end 성능은 baseline 참고용입니다.
- **styleB(취소선)**는 문자 간격을 강제 확보한 합성 조건에서 실용 정확도가 나오도록 설계했습니다. 실제 서비스 캡차는 폰트·간격·배경이 달라 재학습이 필요합니다.
- **모델 파일 크기**: RandomForest 500 trees + HOG는 100~200MB 정도가 됩니다. LinearSVC는 훨씬 작습니다.
- **재현성**: 모든 랜덤에 seed=42를 고정했습니다. 다른 시드로 여러 번 학습해 분산을 관찰하는 것도 실습 포인트.

## 예측 사용법

```python
import joblib
from PIL import Image
import sys
sys.path.insert(0, "training")
from train_digit_model import (
    binarize, remove_hstrike, segment_by_components,
    segment_by_x_clustering, features_of_segments,
)

clf = joblib.load("model.pickle")
im = Image.open("some_6digit_captcha.png")
binv = remove_hstrike(binarize(im), min_len=40)
segs = segment_by_components(binv, 6) or segment_by_x_clustering(binv, 6)
if segs and len(segs) == 6:
    feats = features_of_segments(segs)
    print("".join(clf.predict(feats)))
```
