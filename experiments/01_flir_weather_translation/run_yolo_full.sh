#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# ============================================================
# Full YOLO Gap Establishment Pipeline
# 1. Generate CycleGAN translated images (fog, rain, snow) for test+train
# 2. Set up YOLO synthetic data structure (symlink images, inherit labels)
# 3. Train YOLO 3A models (skip clear_baseline):
#    - fog (cyclediff), fog (cyclegan), rain (cyclegan), snow (cyclegan)
# 4. After each model: evaluate on clear test + adverse test
# ============================================================
set -e
trap '' SIGPIPE

REPO="$ROOT"
PY="python"
CDIR="$REPO/experiments/01_flir_weather_translation"
YOLO="$REPO/data/flir_adas_v2/yolo"
SYNTH="$REPO/data/experiments/01_flir_weather_translation/synthetic"
LOGDIR="$CDIR/logs"
mkdir -p "$LOGDIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir
cd "$REPO"

EPOCHS=100
BATCH=16
IMGSZ=640
WEIGHTS="yolov8s.pt"

echo "=============================================================="
echo " YOLO Gap Establishment — started $(date)"
echo "=============================================================="

# ----------------------------------------------------------------
# Step 0: Verify prerequisites
# ----------------------------------------------------------------
echo ""
echo "=== Step 0: Verify prerequisites ==="

# Check clear YOLO data exists
if [ ! -f "$YOLO/thermal.yaml" ]; then
    echo "ERROR: Clear YOLO data not found. Run: python experiments/01_flir_weather_translation/prepare.py" >&2
    exit 1
fi
echo "  Clear YOLO data: OK"

# Generate video_thermal_test YOLO labels (3749 images) if not present
VTT_LABELS="$YOLO/thermal/labels_vtt_test"
if [ ! -d "$VTT_LABELS" ] || [ "$(ls $VTT_LABELS 2>/dev/null | wc -l)" -lt 3749 ]; then
    echo "  video_thermal_test YOLO labels: generating..."
    python experiments/01_flir_weather_translation/convert_video_thermal_test_yolo.py \
        --output "$VTT_LABELS" 2>&1 || {
        echo "ERROR: Failed to generate video_thermal_test YOLO labels" >&2
        exit 1
    }
    echo "  video_thermal_test YOLO labels: OK ($(ls $VTT_LABELS | wc -l) labels)"
else
    echo "  video_thermal_test YOLO labels: OK ($(ls $VTT_LABELS | wc -l) labels)"
fi

# Check CycleDiff test images
CD_TEST="$SYNTH/cyclediff/fog/test"
if [ ! -d "$CD_TEST" ] || [ "$(ls $CD_TEST 2>/dev/null | wc -l)" -lt 3000 ]; then
    echo "  CycleDiff fog test: MISSING or incomplete"
    echo "  Copying from sample_results..."
    mkdir -p "$CD_TEST"
    # Copy from sample_results if available
    SRC="$REPO/experiments/01_flir_weather_translation/method4_cyclediff/sample_results/clear2fog_model_10_timestep100_v2"
    if [ -d "$SRC" ]; then
        cp "$SRC"/*.png "$CD_TEST/" 2>/dev/null || true
        echo "  Copied $(ls $CD_TEST | wc -l) images"
    else
        echo "  ERROR: No CycleDiff test images found" >&2
        exit 1
    fi
else
    echo "  CycleDiff fog test: OK ($(ls $CD_TEST | wc -l) images)"
fi

# Check CycleDiff train images
CD_TRAIN="$SYNTH/cyclediff/fog/train"
if [ ! -d "$CD_TRAIN" ] || [ "$(ls $CD_TRAIN 2>/dev/null | wc -l)" -lt 10000 ]; then
    echo "  CycleDiff fog train: MISSING or incomplete"
    echo "  (sbatch train translation may still be running)"
    # Check if the train translation output exists
    TRAIN_SRC="$REPO/data/flir_adas_v2/synthetic/cyclediff/fog/train"
    if [ -d "$TRAIN_SRC" ] && [ "$(ls $TRAIN_SRC 2>/dev/null | wc -l)" -gt 10000 ]; then
        echo "  Found train images at $TRAIN_SRC"
    else
        echo "  WARNING: CycleDiff fog train images not ready. 3a cyclediff fog may fail." >&2
    fi
else
    echo "  CycleDiff fog train: OK ($(ls $CD_TRAIN | wc -l) images)"
fi

# ----------------------------------------------------------------
# Step 1: Generate CycleGAN translated images (if not present)
# ----------------------------------------------------------------
echo ""
echo "=== Step 1: Generate CycleGAN translations ==="

for cond in fog rain snow; do
    for split in test train; do
        out_dir="$SYNTH/cyclegan/$cond/$split"
        if [ -d "$out_dir" ] && [ "$(ls $out_dir 2>/dev/null | wc -l)" -gt 100 ]; then
            echo "  cyclegan $cond $split: OK ($(ls $out_dir | wc -l) images)"
            continue
        fi
        echo "  cyclegan $cond $split: generating..."
        mkdir -p "$out_dir"
        cd "$REPO/experiments/01_flir_weather_translation/method3_cyclegan"
        $PY translate.py --config "$cond" --split "$split" --out "$out_dir" \
            2>&1 | tee "$LOGDIR/cyclegan_${cond}_${split}.log"
        rc=${PIPESTATUS[0]}
        cd "$REPO"
        if [ $rc -ne 0 ]; then
            echo "  ERROR: cyclegan $cond $split translation failed ($rc)" >&2
            exit 1
        fi
        echo "  cyclegan $cond $split: done ($(ls $out_dir | wc -l) images)"
    done
done

# ----------------------------------------------------------------
# Step 2: Set up YOLO synthetic data structure
# ----------------------------------------------------------------
echo ""
echo "=== Step 2: Set up YOLO synthetic data ==="

# For each method/condition, create synthetic_{condition}.yaml and inherit labels
for method in cyclediff cyclegan; do
    for cond in fog rain snow; do
        # Skip cyclediff for non-fog (only fog was trained)
        if [ "$method" = "cyclediff" ] && [ "$cond" != "fog" ]; then
            continue
        fi
        adv_root="$SYNTH/$method/$cond"
        if [ ! -d "$adv_root" ]; then
            echo "  SKIP: $adv_root not found"
            continue
        fi
        # Create synthetic_{condition}.yaml
        yaml_path="$adv_root/synthetic_${cond}.yaml"
        if [ ! -f "$yaml_path" ]; then
            cat > "$yaml_path" <<YAML
             path: $adv_root
train: images/train
val: images/val
test: images/test
nc: 15
names: [person, bike, car, motor, bus, train, truck, light, hydrant, sign, face, other_vehicle, backpack, bag, other]
YAML
            echo "  Created $yaml_path"
        fi

        # Create images/ and labels/ subdirs for train/test
        for split in train test; do
            img_dir="$adv_root/images/$split"
            lbl_dir="$adv_root/labels/$split"
            mkdir -p "$img_dir" "$lbl_dir"

            # Copy/symlink translated images if not already there
            src_img_dir="$adv_root/$split"
            if [ -d "$src_img_dir" ] && [ "$(ls $src_img_dir/*.png $src_img_dir/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
                for img in "$src_img_dir"/*.png "$src_img_dir"/*.jpg; do
                    [ -f "$img" ] || continue
                    dst="$img_dir/$(basename $img)"
                    if [ ! -e "$dst" ]; then
                        ln -s "$(realpath $img)" "$dst" 2>/dev/null || cp "$img" "$dst"
                    fi
                done
            fi

            # Inherit labels from clear images (by matching stem)
            if [ "$split" = "train" ]; then
                clear_lbl_dir="$YOLO/thermal/labels/train"
                clear_img_dir="$YOLO/thermal/images/train"
            else
                # Test set uses video_thermal_test labels (3749 images)
                clear_lbl_dir="$VTT_LABELS"
                clear_img_dir="$REPO/data/flir_adas_v2/FLIR_ADAS_v2/video_thermal_test/data"
            fi
            for img_path in "$img_dir"/*; do
                [ -f "$img_path" ] || continue
                stem=$(basename "$img_path" | sed 's/\.[^.]*$//')
                lbl_src="$clear_lbl_dir/${stem}.txt"
                lbl_dst="$lbl_dir/${stem}.txt"
                if [ -f "$lbl_src" ] && [ ! -f "$lbl_dst" ]; then
                    cp "$lbl_src" "$lbl_dst"
                fi
            done
        done
        # Create val as a separate split (first 500 images from test)
        # so YOLO doesn't complain that val resolves to same dir as test
        mkdir -p "$adv_root/images/val"
        first_imgs=$(ls "$adv_root/images/test" 2>/dev/null | head -500)
        for img in $first_imgs; do
            ln -sf "$(realpath "$adv_root/images/test/$img")" "$adv_root/images/val/$img" 2>/dev/null || true
        done
        mkdir -p "$adv_root/labels/val"
        for img in $first_imgs; do
            stem="${img%.*}"
            [ -f "$adv_root/labels/test/${stem}.txt" ] && ln -sf "$(realpath "$adv_root/labels/test/${stem}.txt")" "$adv_root/labels/val/${stem}.txt" 2>/dev/null || true
        done
        echo "  $method/$cond: setup done (train=$(ls $adv_root/images/train 2>/dev/null | wc -l), test=$(ls $adv_root/images/test 2>/dev/null | wc -l))"
    done
done

# Create train_3a directory structure for 3A mode (clear + adverse training)
TRAIN_3A="$YOLO/train_3a"
mkdir -p "$TRAIN_3A/images/train_clear" "$TRAIN_3A/images/train_adv" "$TRAIN_3A/images/val" "$TRAIN_3A/images/test" \
         "$TRAIN_3A/labels/train_clear" "$TRAIN_3A/labels/train_adv" "$TRAIN_3A/labels/val" "$TRAIN_3A/labels/test"

# Symlink clear training images for train_3a
for f in "$YOLO/thermal/images/train"/*; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_3A/images/train_clear/$(basename "$f")"
done
for lbl in "$YOLO/thermal/labels/train"/*; do
    [ -e "$lbl" ] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_3A/labels/train_clear/$(basename "$lbl")"
done

# Symlink clear val/test for train_3a (val -> clear val, test -> clear test)
for f in "$YOLO/thermal/images/val"/*; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_3A/images/val/$(basename "$f")"
done
for lbl in "$YOLO/thermal/labels/val"/*; do
    [ -e "$lbl" ] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_3A/labels/val/$(basename "$lbl")"
done

for f in "$YOLO/thermal/images/test"/*; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$TRAIN_3A/images/test/$(basename "$f")"
done
for lbl in "$YOLO/thermal/labels/test"/*; do
    [ -e "$lbl" ] || continue
    ln -sf "$(realpath "$lbl")" "$TRAIN_3A/labels/test/$(basename "$lbl")"
done

# ----------------------------------------------------------------
# Step 3: Train and evaluate YOLO models
# ----------------------------------------------------------------
echo ""
echo "=== Step 3: Train and evaluate YOLO models ==="

run_gap() {
    local condition=$1
    local method=$2
    local run_name="3a_${method}_${condition}"

    echo ""
    echo "=========================================="
    echo " Training: $run_name"
    echo "=========================================="

    # Set up train_adv symlinks for this method/condition
    ADV_ROOT="$SYNTH/$method/$condition"
    train_adv_imgs="$TRAIN_3A/images/train_adv"
    train_adv_lbls="$TRAIN_3A/labels/train_adv"
    rm -f "$train_adv_imgs"/* "$train_adv_lbls"/* 2>/dev/null
    for f in "$ADV_ROOT"/images/train/*; do
        [ -e "$f" ] || continue
        ln -sf "$(realpath "$f")" "$train_adv_imgs/$(basename "$f")"
    done
    for lbl in "$ADV_ROOT"/labels/train/*; do
        [ -e "$lbl" ] || continue
        ln -sf "$(realpath "$lbl")" "$train_adv_lbls/$(basename "$lbl")"
    done

    # Skip training if model already exists (resume support)
    # YOLO auto-increments run names (e.g., 3a_cyclediff_fog-5), so check for latest
    EXISTING_WEIGHTS=""
    for d in "$CDIR/runs/${run_name}" "$CDIR/runs/"${run_name}-*; do
        if [ -f "$d/weights/best.pt" ]; then
            EXISTING_WEIGHTS="$d/weights/best.pt"
            break
        fi
    done
    if [ -n "$EXISTING_WEIGHTS" ]; then
        echo "--- Training $run_name: SKIPPED (existing weights found at $EXISTING_WEIGHTS)"
    else
        python experiments/01_flir_weather_translation/train_detector.py \
            --mode 3a --condition "$condition" --method "$method" \
            --epochs $EPOCHS --batch-size $BATCH --img-size $IMGSZ \
            --weights "$WEIGHTS" \
            2>&1 | tee "$LOGDIR/${run_name}_train.log"
        TRAIN_RC=${PIPESTATUS[0]}
        if [ $TRAIN_RC -ne 0 ]; then
            echo "ERROR: Training $run_name failed (exit $TRAIN_RC)" >&2
            return 1
        fi
    fi

    # Determine weights path: use existing if training was skipped, else find new run
    if [ -n "$EXISTING_WEIGHTS" ]; then
        WEIGHTS_PATH="$EXISTING_WEIGHTS"
    else
        # YOLO auto-increments run names; find the most recent matching dir
        WEIGHTS_PATH=""
        for d in "$CDIR/runs/${run_name}" "$CDIR/runs/"${run_name}-*; do
            if [ -f "$d/weights/best.pt" ] && [ "$d" -nt "$WEIGHTS_PATH" ] 2>/dev/null; then
                WEIGHTS_PATH="$d/weights/best.pt"
            elif [ -z "$WEIGHTS_PATH" ] && [ -f "$d/weights/best.pt" ]; then
                WEIGHTS_PATH="$d/weights/best.pt"
            fi
        done
        if [ -z "$WEIGHTS_PATH" ]; then
            echo "ERROR: weights not found after training $run_name" >&2
            return 1
        fi
    fi

    echo "--- Evaluating $run_name on clear test ---"
    python experiments/01_flir_weather_translation/eval_detector.py \
        --weights "$WEIGHTS_PATH" \
        --data "data/flir_adas_v2/yolo/thermal.yaml" \
        --name "${run_name}_clear" --split test --imgsz $IMGSZ \
        2>&1 | tee "$LOGDIR/${run_name}_eval_clear.log"

    ADV_DATA="data/flir_adas_v2/synthetic/$method/$condition/synthetic_${condition}.yaml"
    if [ -f "$ADV_DATA" ]; then
        echo "--- Evaluating $run_name on adverse test ---"
        python experiments/01_flir_weather_translation/eval_detector.py \
            --weights "$WEIGHTS_PATH" \
            --data "$ADV_DATA" \
            --name "${run_name}_adv" --split test --imgsz $IMGSZ \
            2>&1 | tee "$LOGDIR/${run_name}_eval_adv.log"
    else
        echo "WARNING: Adverse YAML not found at $ADV_DATA, skipping adverse eval"
    fi

    echo "=== $run_name complete at $(date) ==="
}

# Run gap establishment (skip clear_baseline per user request)
# Continue on failure so subsequent models can still train
run_gap "fog" "cyclediff" || echo "FAILED: fog cyclediff"
run_gap "fog" "cyclegan" || echo "FAILED: fog cyclegan"
run_gap "rain" "cyclegan" || echo "FAILED: rain cyclegan"
run_gap "snow" "cyclegan" || echo "FAILED: snow cyclegan"

echo ""
echo "=============================================================="
echo " All YOLO gap establishment complete at $(date)"
echo "=============================================================="
