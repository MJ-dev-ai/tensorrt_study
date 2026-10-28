# Third-party notices

라이선스 원문 사본: [licenses/](licenses/README.md).

## 적용 범위

이 프로젝트에서 직접 작성하고 별도의 고지가 없는 코드에는 루트의 [Unlicense](LICENSE)가 적용됩니다. 외부 소스, 라이브러리, SDK, 모델 및 데이터에는 각 권리자가 정한 라이선스가 적용됩니다. 루트 LICENSE는 이들의 라이선스를 대체하지 않습니다.

이 문서는 현재 소스에서 확인한 직접 의존성을 안내합니다. 설치된 모든 간접 의존성이나 실행 파일 배포에 필요한 고지의 전체 목록은 아닙니다. 실제 재배포 시에는 사용한 버전에 포함된 LICENSE, NOTICE 및 제3자 고지를 함께 확인하고 필요한 사본을 배포물에 포함해야 합니다.

## 저장소에 포함된 소스: stb

다음 헤더는 Sean Barrett 및 각 헤더에 명시된 기여자가 작성한 외부 코드입니다.

| 파일 | 헤더 버전 | 라이선스 원문 위치 |
|---|---|---|
| [stb_image.h](externals/stb/stb_image.h) | 2.30 | 파일 끝의 ALTERNATIVE A / B |
| [stb_image_write.h](externals/stb/stb_image_write.h) | 1.16 | 파일 끝의 ALTERNATIVE A / B |
| [stb_image_resize2.h](externals/stb/stb_image_resize2.h) | 2.18 | 파일 끝의 ALTERNATIVE A / B |
| [stb_easy_font.h](externals/stb/stb_easy_font.h) | 1.1 | 파일 끝의 ALTERNATIVE A / B |

stb는 MIT 또는 Public Domain (Unlicense) 중 하나를 선택할 수 있는 이중 라이선스입니다. 이 프로젝트에서는 Unlicense 선택지를 사용하며, 헤더에 포함된 저작권 고지와 두 라이선스 원문을 그대로 보존합니다.

- 프로젝트: https://github.com/nothings/stb
- 공식 라이선스: https://github.com/nothings/stb/blob/master/LICENSE

## 별도 설치하는 의존성

다음 패키지와 SDK 자체는 이 Git 저장소에 포함하지 않습니다. import 또는 링크한다고 해당 패키지의 라이선스가 Unlicense로 변경되지는 않습니다.

| 구성 요소 | 사용 위치 | 라이선스 / 공식 안내 |
|---|---|---|
| Ultralytics | python-scripts/onnx_export.py | [AGPL-3.0 또는 Enterprise](https://www.ultralytics.com/license) |
| ONNX | python-scripts/onnx_export.py | [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE), [NOTICE](https://github.com/onnx/onnx/blob/main/NOTICE) |
| ONNX GraphSurgeon | python-scripts/onnx_export.py | [Apache-2.0 소스 고지](https://github.com/NVIDIA/TensorRT/blob/main/tools/onnx-graphsurgeon/onnx_graphsurgeon/ir/graph.py) |
| PySide6 / Qt | python-scripts/nvidia-dmon.py | [Qt 라이선스](https://doc.qt.io/qt-6/licensing.html), [Qt for Python 제3자 고지](https://doc.qt.io/qtforpython-6/licenses.html) |
| nvidia-ml-py | python-scripts/nvidia-dmon.py의 pynvml import | [BSD 계열 라이선스 및 배포 패키지](https://pypi.org/project/nvidia-ml-py/) |
| TensorRT SDK | C++ 엔진 빌드 및 추론 | [NVIDIA TensorRT License Agreement](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html) |
| CUDA Toolkit / Runtime | C++ CUDA 메모리·스트림 처리 | [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) |

PySide6 및 Qt의 적용 조건은 선택한 라이선스와 사용 모듈에 따라 다릅니다. 현재 모니터는 QtCore, QtGui, QtWidgets를 사용합니다. LGPL 방식으로 Qt를 포함하여 배포할 경우 관련 고지, 라이선스 사본, 라이브러리 교체 및 해당 소스 제공 등 적용되는 조건을 충족해야 합니다.

TensorRT SDK 바이너리의 조건과 NVIDIA/TensorRT 저장소의 개별 오픈소스 구성 요소 조건은 구분해야 합니다. 현재 빌드는 TensorRT DLL을 build/bin으로 복사하고 CUDA::cudart_static을 정적 링크합니다. 실행 파일과 DLL을 배포할 때는 사용한 SDK 버전의 재배포 가능 항목과 배포 조건을 확인해야 합니다. SDK 샘플 코드를 복사·수정한 부분이 있다면 그 샘플에 요구되는 고지도 유지해야 합니다.

## 모델 및 데이터

models/의 PT, ONNX, TensorRT engine 파일과 로컬 입력 데이터는 Git에서 제외되며, 프로젝트 Unlicense의 적용 대상으로 선언하지 않습니다.

현재 exporter는 models/hyundai_emblem.pt를 Ultralytics로 읽습니다. 이 모델의 원본 출처, 기반 가중치, 학습 데이터 권리 및 별도 계약 여부는 이 저장소의 코드만으로 확인되지 않았습니다. 모델을 배포할 때는 이러한 정보를 확인하여 모델별 라이선스를 명시해야 합니다.

Ultralytics는 자사 학습 모델에도 기본적으로 AGPL-3.0이 적용된다고 안내합니다. ONNX 또는 TensorRT engine으로 형식을 변환하는 것만으로 원본 모델의 라이선스가 제거되지는 않습니다. 모델과 애플리케이션을 결합해 배포하거나 서비스할 때는 실제 구성에 적용되는 AGPL 조건 또는 별도 Enterprise 계약을 확인해야 합니다. 이 문서의 추가만으로 해당 조건의 충족을 보증하지 않습니다.
