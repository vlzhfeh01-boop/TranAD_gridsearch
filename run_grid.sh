#!/usr/bin/env bash

# ===== 사용자 설정 부분 =====
BASE_CONFIG="./configs/tranad_base.json"   # 템플릿 config
RESULTS_DIR="./results/gridsearch"        # 로그/결과 저장 폴더
DATASET="BAT"                  # --dataset 인자
MODEL="TranAD"                            # --model 인자
BRAND=1                         #brand num
NUM_EPOCHS=5                             # 고정 epoch

# 튜닝할 값들
BATCH_LIST=(64 128)
lr_LIST=(8e-3)

mkdir -p "$RESULTS_DIR"

RESULT_CSV="$RESULTS_DIR/val_auroc_summary.csv"
echo "model,brand,batch,lr,val_auroc,log_path,config_path" >> "$RESULT_CSV"

for w in "${BATCH_LIST[@]}"; do
  for ld in "${lr_LIST[@]}"; do
    cfg_out="./configs/tranad_${BRAND}_batch${w}_lr${ld}.json"
    log_out="${RESULTS_DIR}/tranad_${BRAND}_batch${w}_lr${ld}.log"

    echo "==============================================="
    echo "Running: batch=${w}, lr=${ld}"
    echo "Epochs: $NUM_EPOCHS"
    echo "Config: $cfg_out"
    echo "Log   : $log_out"
    echo "==============================================="

    # 1) base config를 읽어서 n_window, latent_dim, num_epochs 수정 후 새 config로 저장
    python3 - <<EOF
import json

base_path = "${BASE_CONFIG}"
out_path  = "${cfg_out}"

with open(base_path, "r") as f:
    cfg = json.load(f)

cfg["training"]["optimizer"]["lr"] = ${ld}
cfg["training"]["batch_size"] = ${w}

with open(out_path, "w") as f:
    json.dump(cfg, f, indent=4)
EOF

    # 2) main.py 실행 (VAL_AUROC를 로그에 찍게 되어 있어야 함)
    python3 main.py \
        --config "$cfg_out" \
        --dataset "$DATASET" \
        --model "$MODEL" 2>&1 | tee "$log_out"

    # 3) 로그에서 VAL_AUROC=... 라인 파싱
    val_auroc=$(grep "AUROC" "$log_out" | tail -n 1 | awk -F'=' '{print $2}')

    if [ -z "$val_auroc" ]; then
      val_auroc="NaN"
    fi

    echo "${MODEL},${BRAND},${w},${ld},${val_auroc},${log_out},${cfg_out}" >> "$RESULT_CSV"
  done
done

echo "==============================================="
echo "완료! 결과 요약: $RESULT_CSV"
cat "$RESULT_CSV"

