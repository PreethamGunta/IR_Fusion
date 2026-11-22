#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# ============================================================
# Fog-Specific Pipeline
# 
# 1. Translate CycleGAN foggy train images
# 2. Create 75% strength blended foggy images (cyclediff fog + 25% clear)
#    Uses yolo/thermal/images/train (9051) to AVOID val contamination
# 3. Train YOLOv8s 3A (clear + 75% foggy blend) and evaluate clear vs foggy
# 4. Delete + redo CycleGAN 3A fog training + evaluation
# 
# Training note: YOLO treats train: [train_clear, train_adv] as a flat
# concatenation. Batch=16 means 16 random images from the combined pool
# (~19793 images: 9051 clear + 10742 adverse). No per-batch balancing.
# ============================================================
set -e
trap '' SIGPIPE

REPO="$ROOT"
PY="python"
CDIR="$REPO/experiments/01_flir_weather_translation"
YOLO="$REPO/data/flir_adas_v2/yolo"
SYNTH="$REPO/data/flir_adas_v2/synthetic"
LOGDIR="$CDIR/logs/fog_pipeline"
mkdir -p "$LOGDIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir
cd "$REPO"

EPOCHS=100
BATCH=16
IMGSZ=640
WEIGHTS="yolov8s.pt"
STRENGTH=0.75
CLEAR_LABELS="$YOLO/thermal/labels/train"
TRAIN_3A="$YOLO/train_3a"
TRAIN_ADV_IMGS="$TRAIN_3A/images/train_adv"
TRAIN_ADV_LBLS="$TRAIN_3A/labels/train_adv"

echo "=============================================================="
echo " Fog-Specific Pipeline — started $(date)"
echo " Blend: ${STRENGTH} foggy + 0.25 clear"
echo " Batch=$BATCH (flat concatenation of clear + adverse, no per-batch balancing)"
echo " GPU: $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | head -1)"
echo "=============================================================="

# ----------------------------------------------------------------
# Step 1: Translate CycleGAN foggy train images
# ----------------------------------------------------------------
echo ""
echo "=== Step 1: Translate CycleGAN fog train ==="
FOG_TRAIN_DIR="$SYNTH/cyclegan/fog/train"
if [ -d "$FOG_TRAIN_DIR" ] && [ "$(ls "$FOG_TRAIN_DIR"/*.jpg 2>/dev/null | wc -l)" -gt 5000 ]; then
    echo "  cyclegan fog train: already populated ($(ls "$FOG_TRAIN_DIR"/*.jpg 2>/dev/null | wc -l) images)"
else
    echo "  cyclegan fog train: generating..."
    mkdir -p "$FOG_TRAIN_DIR"
    cd "$CDIR/method3_cyclegan"
    $PY translate.py --config fog --split train --out "$FOG_TRAIN_DIR" \
        2>&1 | tee "$LOGDIR/cyclegan_fog_train_translate.log"
    rc=${PIPESTATUS[0]}
    cd "$REPO"
    if [ $rc -ne 0 ]; then
        echo "ERROR: cyclegan fog train translation failed ($rc)" >&2
        exit 1
    fi
    echo "  cyclegan fog train: done ($(ls "$FOG_TRAIN_DIR"/*.jpg 2>/dev/null | wc -l) images)"
fi

# ----------------------------------------------------------------
# Step 2: Blend Cyclediff fog at 75% strength (train + test)
# Using yolo/thermal/images/train as clear source to avoid val contamination
# ----------------------------------------------------------------
echo ""
echo "=== Step 2: Blend Cyclediff fog at ${STRENGTH} strength ==="
BLEND_DIR="$SYNTH/cyclediff_75fog"

for split in train test; do
    OUT_DIR="$BLEND_DIR/fog/$split"
    EXPECTED=100
    if [ "$split" = "train" ]; then
        EXPECTED=5000
    fi
    
    if [ -d "$OUT_DIR" ] && [ "$(ls "$OUT_DIR"/*.jpg 2>/dev/null | wc -l)" -gt $EXPECTED ]; then
        echo "  blend fog $split: already done ($(ls "$OUT_DIR"/*.jpg 2>/dev/null | wc -l) images)"
    else
        echo "  blend fog $split: generating..."
        rm -rf "$OUT_DIR"
        $PY "$CDIR/blend_cyclediff_fog.py" \
            --split "$split" --strength $STRENGTH \
            2>&1 | tee "$LOGDIR/blend_fog_${split}.log"
        echo "  blend fog $split: done ($(ls "$OUT_DIR"/*.jpg 2>/dev/null | wc -l) images)"
    fi
done

# ----------------------------------------------------------------
# Step 3: Create 75% fog synthetic YAML, images/ symlinks, and labels
# ----------------------------------------------------------------
echo ""
echo "=== Step 3: Create 75% fog synthetic YAML ==="
BLEND_FOG="$BLEND_DIR/fog"

# Create val directory and all labels directories (train/test become symlinks)
mkdir -p "$BLEND_FOG/images/val" "$BLEND_FOG/labels/train" "$BLEND_FOG/labels/val" "$BLEND_FOG/labels/test"

# Create file-level symlinks for images/train and images/test (NOT directory symlinks,
# because YOLO follows directory symlinks and loses label path resolution)
mkdir -p "$BLEND_FOG/images/test" "$BLEND_FOG/images/train"
find "$BLEND_FOG/images/test" -mindepth 1 -delete 2>/dev/null || true
find "$BLEND_FOG/images/train" -mindepth 1 -delete 2>/dev/null || true
for f in "$BLEND_FOG/test"/*.jpg; do
    [ -f "$f" ] || continue
    ln -sf "$(realpath "$f")" "$BLEND_FOG/images/test/$(basename "$f")"
done
for f in "$BLEND_FOG/train"/*.jpg; do
    [ -f "$f" ] || continue
    ln -sf "$(realpath "$f")" "$BLEND_FOG/images/train/$(basename "$f")"
done

# Inherit test labels from VTT test labels (3749 images)
VTT_LABELS="$YOLO/thermal/labels_vtt_test"
for img in "$BLEND_FOG/test"/*; do
    [ -f "$img" ] || continue
    [[ "$img" == *.jpg ]] || continue
    stem=$(basename "$img" .jpg)
    lbl="$VTT_LABELS/${stem}.txt"
    if [ -f "$lbl" ] && [ ! -f "$BLEND_FOG/labels/test/${stem}.txt" ]; then
        cp "$lbl" "$BLEND_FOG/labels/test/${stem}.txt"
    fi
done

# Create val as first 500 test images (matching cyclegan convention)
VAL_DIR="$BLEND_FOG/images/val"
VAL_LBL="$BLEND_FOG/labels/val"
find "$VAL_DIR" -mindepth 1 -delete 2>/dev/null || true
find "$VAL_LBL" -mindepth 1 -delete 2>/dev/null || true
first_imgs=$(ls "$BLEND_FOG/images/test" 2>/dev/null | head -500)
for img in $first_imgs; do
    ln -sf "$(realpath "$BLEND_FOG/images/test/$img")" "$VAL_DIR/$img" 2>/dev/null || true
    stem="${img%.*}"
    [ -f "$BLEND_FOG/labels/test/${stem}.txt" ] && ln -sf "$(realpath "$BLEND_FOG/labels/test/${stem}.txt")" "$VAL_LBL/${stem}.txt" 2>/dev/null || true
done

# Inherit train labels from clear thermal train
for img in "$BLEND_FOG/train"/*; do
    [ -f "$img" ] || continue
    [[ "$img" == *.jpg ]] || continue
    stem=$(basename "$img" .jpg)
    lbl="$CLEAR_LABELS/${stem}.txt"
    if [ -f "$lbl" ] && [ ! -f "$BLEND_FOG/labels/train/${stem}.txt" ]; then
        cp "$lbl" "$BLEND_FOG/labels/train/${stem}.txt"
    fi
done

cat > "$BLEND_FOG/synthetic_fog.yaml" << YAML
path: $BLEND_FOG
train: images/train
val: images/val
test: images/test
nc: 15
names: [person, bike, car, motor, bus, train, truck, light, hydrant, sign, face, other_vehicle, backpack, bag, other]
YAML

BLEND_TEST_COUNT=$(ls "$BLEND_FOG/images/test"/*.jpg 2>/dev/null | wc -l)
BLEND_TRAIN_COUNT=$(ls "$BLEND_FOG/images/train"/*.jpg 2>/dev/null | wc -l)
BLEND_VAL_COUNT=$(ls "$BLEND_FOG/images/val"/*.jpg 2>/dev/null | wc -l)
echo "  75% fog: train=$BLEND_TRAIN_COUNT, val=$BLEND_VAL_COUNT, test=$BLEND_TEST_COUNT"

# ----------------------------------------------------------------
# Step 4: Set up train_3a for 75% foggy training
# ----------------------------------------------------------------
echo ""
echo "=== Step 4: Set up train_3a for 75% foggy training ==="

# Create all required train_3a subdirectories
mkdir -p "$TRAIN_3A/images/train_clear" "$TRAIN_3A/images/train_adv" \
         "$TRAIN_3A/images/val" "$TRAIN_3A/images/test" \
         "$TRAIN_3A/labels/train_clear" "$TRAIN_3A/labels/train_adv" \
         "$TRAIN_3A/labels/val" "$TRAIN_3A/labels/test"

# Set up clear training images/labels (train_clear symlinks)
echo "  Setting up train_clear symlinks (clear thermal train)..."
find "$TRAIN_3A/images/train_clear" -mindepth 1 -delete 2>/dev/null || true
find "$TRAIN_3A/labels/train_clear" -mindepth 1 -delete 2>/dev/null || true
for f in "$YOLO/thermal/images/train"/*; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_3A/images/train_clear/$(basename "$f")"
done
for lbl in "$CLEAR_LABELS"/*; do
    [ -e "$lbl" ] || continue
    [[ "$lbl" == *.txt ]] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_3A/labels/train_clear/$(basename "$lbl")"
done
CLEAR_COUNT=$(ls "$TRAIN_3A/images/train_clear"/*.jpg 2>/dev/null | wc -l)
echo "  train_clear: $CLEAR_COUNT images"

# Set up val/test from clear thermal (val/test for training validation and eval)
echo "  Setting up val/test symlinks (clear thermal)..."
find "$TRAIN_3A/images/val" -mindepth 1 -delete 2>/dev/null || true
find "$TRAIN_3A/labels/val" -mindepth 1 -delete 2>/dev/null || true
for f in "$YOLO/thermal/images/val"/*; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_3A/images/val/$(basename "$f")"
done
for lbl in "$YOLO/thermal/labels/val"/*; do
    [ -e "$lbl" ] || continue
    [[ "$lbl" == *.txt ]] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_3A/labels/val/$(basename "$lbl")"
done
find "$TRAIN_3A/images/test" -mindepth 1 -delete 2>/dev/null || true
find "$TRAIN_3A/labels/test" -mindepth 1 -delete 2>/dev/null || true
for f in "$YOLO/thermal/images/test"/*; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_3A/images/test/$(basename "$f")"
done
for lbl in "$YOLO/thermal/labels/test"/*; do
    [ -e "$lbl" ] || continue
    [[ "$lbl" == *.txt ]] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_3A/labels/test/$(basename "$lbl")"
done

# Set up train_adv symlinks (75% blend foggy)
echo "  Setting up train_adv symlinks (75% blend foggy)..."
find "$TRAIN_ADV_IMGS" -mindepth 1 -delete 2>/dev/null || true
find "$TRAIN_ADV_LBLS" -mindepth 1 -delete 2>/dev/null || true
for f in "$BLEND_FOG/images/train"/*; do
    [ -e "$f" ] || continue
    [[ "$f" == *.jpg ]] || [ "$f" == *.png ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_ADV_IMGS/$(basename "$f")"
done
for lbl in "$BLEND_FOG/labels/train"/*; do
    [ -e "$lbl" ] || continue
    [[ "$lbl" == *.txt ]] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_ADV_LBLS/$(basename "$lbl")"
done
ADV_COUNT=$(ls "$TRAIN_ADV_IMGS"/*.jpg 2>/dev/null | wc -l)
ADV_LBL_COUNT=$(ls "$TRAIN_ADV_LBLS"/*.txt 2>/dev/null | wc -l)
echo "  train_adv: $ADV_COUNT images, $ADV_LBL_COUNT labels"

# ----------------------------------------------------------------
# Step 5: Train 3A YOLOv8s on clear + 75% foggy (cyclediff blend)
# ----------------------------------------------------------------
echo ""
echo "=== Step 5: Train 3A cyclediff_fog_75 (clear + 75% blend foggy) ==="
RUN_NAME="3a_cyclediff_fog"
EXISTING_WEIGHTS=""
for d in "$CDIR/runs/${RUN_NAME}" "$CDIR/runs/"${RUN_NAME}-*; do
    if [ -f "$d/weights/best.pt" ]; then
        EXISTING_WEIGHTS="$d/weights/best.pt"
        break
    fi
done

if [ -n "$EXISTING_WEIGHTS" ]; then
    echo "  $RUN_NAME: SKIPPED (existing weights at $EXISTING_WEIGHTS)"
    WEIGHTS_PATH="$EXISTING_WEIGHTS"
else
    echo "  Training $RUN_NAME (${EPOCHS} epochs, batch=$BATCH, imgsz=$IMGSZ)..."
    CUDA_VISIBLE_DEVICES=0 $PY "$CDIR/train_detector.py" \
        --mode 3a --condition fog --method cyclediff \
        --epochs $EPOCHS --batch-size $BATCH --img-size $IMGSZ \
        --weights "$WEIGHTS" \
        2>&1 | tee "$LOGDIR/${RUN_NAME}_train.log"
    TRAIN_RC=${PIPESTATUS[0]}
    if [ $TRAIN_RC -ne 0 ]; then
        echo "ERROR: Training $RUN_NAME failed (exit $TRAIN_RC)" >&2
        exit 1
    fi
    WEIGHTS_PATH=""
    for d in "$CDIR/runs/${RUN_NAME}" "$CDIR/runs/"${RUN_NAME}-*; do
        if [ -f "$d/weights/best.pt" ] && { [ -z "$WEIGHTS_PATH" ] || [ "$d" -nt "$WEIGHTS_PATH" ]; }; then
            WEIGHTS_PATH="$d/weights/best.pt"
        fi
    done
    [ -z "$WEIGHTS_PATH" ] && { echo "ERROR: no weights found for $RUN_NAME" >&2; exit 1; }
fi
echo "  $RUN_NAME weights: $WEIGHTS_PATH"

# ----------------------------------------------------------------
# Step 6: Evaluate 3A cyclediff_fog_75
# ----------------------------------------------------------------
echo ""
echo "=== Step 6: Evaluate $RUN_NAME ==="

echo "  Evaluating on clear test..."
$PY "$CDIR/eval_detector.py" \
    --weights "$WEIGHTS_PATH" \
    --data "$YOLO/thermal.yaml" \
    --name "${RUN_NAME}_clear" --split test --imgsz $IMGSZ \
    --force \
    2>&1 | tee "$LOGDIR/${RUN_NAME}_eval_clear.log" || echo "WARN: clear eval failed"

echo "  Evaluating on 75% foggy test..."
$PY "$CDIR/eval_detector.py" \
    --weights "$WEIGHTS_PATH" \
    --data "$BLEND_FOG/synthetic_fog.yaml" \
    --name "${RUN_NAME}_adv" --split test --imgsz $IMGSZ \
    --force \
    2>&1 | tee "$LOGDIR/${RUN_NAME}_eval_adv.log" || echo "WARN: foggy eval failed"

# ----------------------------------------------------------------
# Step 7: Delete old cyclegan fog results + redo training + eval
# ----------------------------------------------------------------
echo ""
echo "=== Step 7: Redo 3A cyclegan fog (delete old, retrain, re-eval) ==="
RUN_NAME_CG="3a_cyclegan_fog"

# Delete old cyclegan fog run directories (user requested "delete and redo")
echo "  Deleting old cyclegan fog runs and results..."
rm -rf "$CDIR/runs/${RUN_NAME_CG}" "$CDIR/runs/"${RUN_NAME_CG}-* 2>/dev/null || true
rm -f "$CDIR/results/${RUN_NAME_CG}_clear.json" "$CDIR/results/${RUN_NAME_CG}_adv.json" 2>/dev/null || true
rm -rf "$CDIR/runs/eval_${RUN_NAME_CG}_clear" "$CDIR/runs/eval_${RUN_NAME_CG}_adv" 2>/dev/null || true

# Verify cyclegan fog train images present
CG_TRAIN_COUNT=$(ls "$SYNTH/cyclegan/fog/images/train/"*.jpg 2>/dev/null | wc -l)
echo "  cyclegan fog train images: $CG_TRAIN_COUNT"
[ "$CG_TRAIN_COUNT" -lt 5000 ] && { echo "ERROR: insufficient cyclegan fog train images" >&2; exit 1; }

# Set up train_adv symlinks with cyclegan fog (100% foggy)
# Filter: only include images whose stems exist in train_clear (avoid val contamination)
echo "  Setting up train_adv symlinks (100% cyclegan fog, train-only filter)..."
find "$TRAIN_ADV_IMGS" -mindepth 1 -delete 2>/dev/null || true
find "$TRAIN_ADV_LBLS" -mindepth 1 -delete 2>/dev/null || true
for f in "$SYNTH/cyclegan/fog/images/train"/*; do
    [ -e "$f" ] || continue
    [[ "$f" == *.jpg ]] || continue
    stem=$(basename "$f" .jpg)
    # Only include if this stem exists in train_clear (filters out val/test images)
    if [ -e "$YOLO/thermal/images/train/${stem}.jpg" ]; then
        ln -sf "$(realpath "$f")" "$TRAIN_ADV_IMGS/$(basename "$f")"
    fi
done
for lbl in "$SYNTH/cyclegan/fog/labels/train"/*; do
    [ -e "$lbl" ] || continue
    [[ "$lbl" == *.txt ]] || continue
    stem=$(basename "$lbl" .txt)
    if [ -e "$YOLO/thermal/images/train/${stem}.jpg" ]; then
        ln -sf "$(realpath "$lbl")" "$TRAIN_ADV_LBLS/$(basename "$lbl")"
    fi
done
ADV_COUNT_CG=$(ls "$TRAIN_ADV_IMGS"/*.jpg 2>/dev/null | wc -l)
ADV_LBL_CG=$(ls "$TRAIN_ADV_LBLS"/*.txt 2>/dev/null | wc -l)
echo "  train_adv: $ADV_COUNT_CG images, $ADV_LBL_CG labels"

echo "  Training $RUN_NAME_CG (${EPOCHS} epochs, batch=$BATCH, imgsz=$IMGSZ)..."
CUDA_VISIBLE_DEVICES=0 $PY "$CDIR/train_detector.py" \
    --mode 3a --condition fog --method cyclegan \
    --epochs $EPOCHS --batch-size $BATCH --img-size $IMGSZ \
    --weights "$WEIGHTS" \
    2>&1 | tee "$LOGDIR/${RUN_NAME_CG}_train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [ $TRAIN_RC -ne 0 ]; then
    echo "ERROR: Training $RUN_NAME_CG failed (exit $TRAIN_RC)" >&2
    exit 1
fi
WEIGHTS_PATH_CG=""
for d in "$CDIR/runs/${RUN_NAME_CG}" "$CDIR/runs/"${RUN_NAME_CG}-*; do
    if [ -f "$d/weights/best.pt" ] && { [ -z "$WEIGHTS_PATH_CG" ] || [ "$d" -nt "$WEIGHTS_PATH_CG" ]; }; then
        WEIGHTS_PATH_CG="$d/weights/best.pt"
    fi
done
[ -z "$WEIGHTS_PATH_CG" ] && { echo "ERROR: no weights found for $RUN_NAME_CG" >&2; exit 1; }
echo "  $RUN_NAME_CG weights: $WEIGHTS_PATH_CG"

echo "  Evaluating $RUN_NAME_CG on clear test..."
$PY "$CDIR/eval_detector.py" \
    --weights "$WEIGHTS_PATH_CG" \
    --data "$YOLO/thermal.yaml" \
    --name "${RUN_NAME_CG}_clear" --split test --imgsz $IMGSZ \
    --force \
    2>&1 | tee "$LOGDIR/${RUN_NAME_CG}_eval_clear.log" || echo "WARN: clear eval failed"

echo "  Evaluating $RUN_NAME_CG on 100% cyclegan foggy test..."
$PY "$CDIR/eval_detector.py" \
    --weights "$WEIGHTS_PATH_CG" \
    --data "$SYNTH/cyclegan/fog/synthetic_fog.yaml" \
    --name "${RUN_NAME_CG}_adv" --split test --imgsz $IMGSZ \
    --force \
    2>&1 | tee "$LOGDIR/${RUN_NAME_CG}_eval_adv.log" || echo "WARN: foggy eval failed"

echo ""
echo "=============================================================="
echo " Fog pipeline complete at $(date)"
echo "=============================================================="
