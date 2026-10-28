# tensorrt_study

YOLO11 검출 모델을 ONNX에서 TensorRT 엔진으로 빌드하고, C++로 이미지 추론과 결과 시각화를 수행하는 학습 프로젝트입니다.

## 폴더 구조

```text
tensorrt_study/
├── CMakeLists.txt
├── Makefile
├── README.md
├── LICENSE
├── src/
│   ├── inference_once.cpp       # 모델 로딩·추론·이미지 입출력
│   └── cv/
│       ├── cv.hpp              # CvImage와 이미지 처리 API
│       └── cv.cpp              # 그리기·텍스트·리사이즈·검출 후처리
├── externals/
│   └── stb/                    # 이미지 입출력·글꼴·리사이즈 헤더
├── python-scripts/
│   ├── onnx_export.py          # PT → ONNX 내보내기
│   └── nvidia-dmon.py          # GPU 사용률 모니터 GUI
├── models/                    # 로컬 모델 파일: PT, ONNX, engine
├── data/
│   ├── inputs/                # 입력 이미지
│   └── outputs/               # 추론 결과 이미지
├── outputs/
│   ├── logs/                  # 날짜별 TensorRT 로그
│   └── profiles/              # 프로파일 결과 보관용
├── build/
│   └── bin/                   # 실행 파일과 TensorRT DLL
└── TensorRT-.../               # 로컬 SDK; 외부 경로로 지정해도 됨
```

`models`, `data/inputs`, `data/outputs`, `outputs`, `build`, SDK 및 압축 파일은 `.gitignore`로 제외합니다. 저장소를 새로 받으면 모델·입력 데이터·SDK를 별도로 준비해야 합니다. 로컬 `include/`는 현재 빌드에서 사용하지 않습니다.

## 준비 환경

- Windows x64
- Visual Studio 2022 또는 Build Tools: C++ 데스크톱 개발 도구와 Windows SDK
- CMake 3.24 이상, C++17
- NVIDIA GPU와 드라이버, CUDA Toolkit, TensorRT SDK
- 선택: GNU Make, Python 환경

TensorRT와 CUDA는 사용하는 SDK 및 GPU에 맞는 조합을 준비하세요. 현재 CMake는 TensorRT 11의 `nvinfer_11`, `nvonnxparser_11` 라이브러리를 우선 검색합니다.

SDK 기본 검색 위치는 다음과 같습니다.

```text
TensorRT-11.2.1.2-Windows-amd64-cuda-13.3/TensorRT-11.2.1.2/
```

## 1. inference_once.cpp

`src/inference_once.cpp`는 이미지 로딩부터 TensorRT 추론, 박스·라벨 표시와 결과 저장까지 수행하는 예제입니다. 빌드 타깃은 `inference_once`입니다.

현재 예제는 FP32 입력 `[N, 3, 2048, 2048]`, 출력 `[N, 8, 86016]`인 **4개 클래스 모델**을 대상으로 합니다.

### 빌드

아래 명령은 프로젝트 루트에서 실행합니다. 명령 예시는 Git Bash 기준이며, CMake 명령은 PowerShell에서도 사용할 수 있습니다.

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
```

SDK를 다른 위치에 설치했다면 구성 단계에서 지정합니다. 다음 경로는 실제 설치 위치로 바꾸세요.

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
  -DTENSORRT_ROOT="D:/sdk/TensorRT" \
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y"
```

빌드 결과는 `build/bin/inference_once.exe`이며 필요한 TensorRT DLL도 같은 폴더로 복사됩니다. 루트에 예전 `inference_once.exe`가 남아 있더라도 새 실행 파일을 사용하세요.

#### Makefile 사용

GNU Make가 설치되어 있으면 다음 명령도 가능합니다. 환경에 따라 `make` 대신 `mingw32-make`를 사용합니다.

```bash
make build
make run RUN_ARGS="--batch 1 data/inputs/normal_map_1.png"
make clean
```

`make run`만 입력하면 빌드 후 `--help`를 실행합니다. `clean`은 빌드 산출물을 정리하며 모델과 데이터는 삭제하지 않습니다.

```bash
make build CONFIG=Debug
make build CMAKE_ARGS='-DTENSORRT_ROOT="D:/sdk/TensorRT"'
```

`BUILD_DIR`, `CONFIG`, `GENERATOR`, `ARCH`, `PYTHON`, `CMAKE_ARGS`, `RUN_ARGS`를 재정의할 수 있습니다. SDK 설정은 CMake 캐시에 유지됩니다.

### 모델·입력 준비

기본 파일명은 배치 크기 `N`을 반영합니다.

```text
models/hyundai_emblem_bN_fp32.engine
models/hyundai_emblem_bN_fp32.onnx
```

엔진이 있으면 로드하고, 없으면 ONNX로 엔진을 빌드해 저장합니다. 이미지 인자 없이 실행하면 엔진 준비까지만 수행합니다. 고정 배치 ONNX/엔진은 `--batch`와 배치 크기가 일치해야 합니다.

```bash
./build/bin/inference_once.exe --batch 1
```

### 추론 실행

#### 이미지 한 장

```bash
./build/bin/inference_once.exe --batch 1 data/inputs/normal_map_1.png
```

#### 서로 다른 이미지 네 장

```bash
./build/bin/inference_once.exe --batch 4 \
  data/inputs/normal_map_1.png \
  data/inputs/normal_map_2.png \
  data/inputs/normal_map_3.png \
  data/inputs/normal_map_4.png
```

배치가 4일 때 이미지 한 장만 전달하면 같은 이미지를 네 번 사용합니다. 입력 이미지 수는 **1개 또는 배치 크기와 같은 수**여야 합니다.

#### 모델과 출력 폴더 직접 지정

```bash
./build/bin/inference_once.exe --batch 1 \
  --onnx models/custom.onnx \
  --engine models/custom.engine \
  --output-dir data/outputs/custom \
  data/inputs/normal_map_1.png
```

| 인자 | 기본값 / 동작 |
|---|---|
| `--batch N` | 기본값 `4`, 양의 정수 |
| `--onnx PATH` | `models/hyundai_emblem_bN_fp32.onnx` |
| `--engine PATH` | `models/hyundai_emblem_bN_fp32.engine` |
| `--output-dir DIR` | `data/outputs/` |
| `--help` | 사용법 표시 |
| 위치 인자 | 이미지 경로 목록 |

기본 모델·출력·로그 경로는 **빌드 시 지정된 프로젝트 루트** 기준입니다. 직접 전달하는 상대 경로는 **현재 작업 디렉터리** 기준입니다. 프로젝트를 이동하면 CMake 구성과 빌드를 다시 진행하세요. 공백이 있는 경로는 큰따옴표로 감쌉니다.

### 결과

- 배치 1: `data/outputs/output.png`
- 여러 배치: `data/outputs/output_0.png`, `output_1.png`, …
- 로그: `outputs/logs/YY-MM-DD/log.txt`
- 박스 라벨: `class_id:score` (예: `1:0.92`)

같은 출력 경로로 재실행하면 이미지 파일을 덮어씁니다. 별도로 보관하려면 `--output-dir`을 지정하세요.

전처리는 RGB를 정규화된 float CHW로 변환한 뒤 `cv::resize()`를 사용합니다. 후처리는 `cv::drawBbox()`에서 신뢰도 필터와 NMS를 적용하고 박스·라벨을 그립니다. 이미지는 패딩 없이 입력 크기로 리사이즈합니다.

### 실행 문제

- **`nvonnxparser_11.dll`을 찾지 못함**: 루트의 이전 실행 파일 대신 `build/bin/inference_once.exe`를 실행하고, 같은 폴더에 DLL이 있는지 확인합니다.
- **모델을 찾지 못함**: 모델은 Git에 포함되지 않습니다. `models/`에 준비하거나 `--onnx`, `--engine`을 지정합니다.
- **배치/출력 shape 불일치**: 모델의 고정 배치와 `--batch`를 맞추고, 현재 예제가 요구하는 4개 클래스·2048 입력 모델인지 확인합니다.

## Python 스크립트

### ONNX 내보내기

필요한 Python 패키지는 `ultralytics`, `onnx`, `onnx_graphsurgeon`입니다. 프로젝트에서 사용하는 export 옵션을 지원하는 환경에서 실행하세요.

```bash
python python-scripts/onnx_export.py
# 또는
make export
```

- 입력: `models/hyundai_emblem.pt`
- 출력: 원본 PT와 같은 `models/` 폴더에 exporter가 반환하는 ONNX 경로
- 현재 설정: 배치 `1`, 이미지 크기 `2048`, 고정 shape, opset `17`, NMS 제외
- 배치와 export 옵션은 스크립트의 `main()`에서 변경합니다. CLI 옵션은 없습니다.

스크립트는 ONNX 그래프 정렬과 검증 후 저장합니다. 자동으로 C++ 예제의 기본 모델명으로 바꾸지는 않으므로, 출력된 경로를 `--onnx`에 전달하거나 기본 파일명에 맞춰 준비하세요. `quantize=16`을 포함한 export 옵션의 지원 여부는 설치된 Ultralytics 환경에 따라 확인해야 합니다.

### GPU 모니터

```bash
python -m pip install PySide6 nvidia-ml-py
python python-scripts/nvidia-dmon.py --gpu 0 --capacity 180 --interval 1
```

GPU 없이 GUI를 확인하려면:

```bash
python python-scripts/nvidia-dmon.py --demo
```

`make monitor`로도 실행할 수 있습니다. Python 3.10 이상이 필요하며, 기록은 메모리에만 유지되고 창을 닫으면 사라집니다. 모니터는 실행 중 자동으로 로그 파일을 저장하지 않습니다.
