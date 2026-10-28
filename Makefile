# GNU Make wrapper; paths are relative to this Makefile.
PROJECT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
CMAKE ?= cmake
BUILD_DIR ?= $(PROJECT_DIR)/build
CONFIG ?= Release
GENERATOR ?= Visual Studio 17 2022
ARCH ?= x64
PYTHON ?= python
CMAKE_ARGS ?=
RUN_ARGS ?= --help

.PHONY: all configure build clean run export monitor

all: build

configure:
	"$(CMAKE)" -S "$(PROJECT_DIR)" -B "$(BUILD_DIR)" -G "$(GENERATOR)" -A "$(ARCH)" $(CMAKE_ARGS)

build: configure
	"$(CMAKE)" --build "$(BUILD_DIR)" --config "$(CONFIG)" --parallel

clean:
	"$(CMAKE)" --build "$(BUILD_DIR)" --config "$(CONFIG)" --target clean

run: build
	"$(BUILD_DIR)/bin/inference_once.exe" $(RUN_ARGS)

export:
	"$(PYTHON)" "$(PROJECT_DIR)/python-scripts/onnx_export.py"

monitor:
	"$(PYTHON)" "$(PROJECT_DIR)/python-scripts/nvidia-dmon.py"
