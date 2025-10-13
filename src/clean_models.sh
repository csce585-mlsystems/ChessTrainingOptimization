#!/bin/bash
# ───────────────────────────────────────────────
# Jarvis cleanup utility: deletes model artifacts
# ───────────────────────────────────────────────

TARGETS=("replay_buffer_v2.npz" "value_net.onnx" "model.pt")

echo "🧹 Initiating cleanup in $(pwd)..."

for file in "${TARGETS[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "✅ Deleted: $file"
    else
        echo "⚪ Skipped (not found): $file"
    fi
done

echo "✨ Cleanup complete, sir."
