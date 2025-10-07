# =============================
# ⚙️  Compiler Config
# =============================
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC

# =============================
# 🐍 Python Integration
# =============================
PY_INCLUDES := $(shell python3 -m pybind11 --includes)
PY_EXT := $(shell python3-config --extension-suffix)

# =============================
# 🧠 ONNX Runtime GPU Paths
# =============================
ONNX_PATH := ./onnxruntime-linux-x64-gpu-1.23.0
ONNX_INCLUDE := -I$(ONNX_PATH)/include
ONNX_LIB := -L$(ONNX_PATH)/lib
ONNX_LIBS := -lonnxruntime

# =============================
# 📁 Sources & Target
# =============================
SRC := chessbridge.cpp chess_engine.cpp game_manager.cpp
TARGET := chessbridge$(PY_EXT)

# 🧠 Add RPATH so runtime finds GPU libs automatically
RPATH := -Wl,-rpath,$(abspath $(ONNX_PATH))/lib

# =============================
# 🧠 Build Target
# =============================
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) \
	$(PY_INCLUDES) $(ONNX_INCLUDE) $(ONNX_LIB) $(ONNX_LIBS) $(RPATH)

# =============================
# 🧹 Clean
# =============================
clean:
	rm -f $(TARGET)
