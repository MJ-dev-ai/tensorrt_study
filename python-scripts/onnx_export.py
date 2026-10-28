import onnx
from ultralytics import YOLO
from pathlib import Path
import onnx_graphsurgeon as gs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

def main():
    """Export models/hyundai_emblem.pt beside its source, independent of cwd."""
    # Load a model
    batch_size = 1
    model_path = MODEL_DIR / "hyundai_emblem.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(str(model_path))

    # Export the model to ONNX format
    output_path = model.export(
        format="onnx",
        imgsz=2048,
        batch=batch_size,
        dynamic=False,
        simplify=False,
        opset=17,
        device="cpu",
        quantize=16,
        nms=False
    )
    exported_model = onnx.load(output_path)
    graph = gs.import_onnx(exported_model)
    graph.toposort()

    sorted_model = gs.export_onnx(graph)
    exported_model.graph.CopyFrom(sorted_model.graph)
    onnx.checker.check_model(exported_model)
    onnx.save(exported_model, output_path)
    print(f"Model exported to ONNX format and saved at: {output_path}")

if __name__ == "__main__":
    main()