#!/usr/bin/env bash

# ===== 사용자 설정 부분 =====
BASE_CONFIG="./configs/tranad_base.json"   # 템플릿 config
RESULTS_DIR="./results/gridsearch"        # 로그/결과 저장 폴더
DATASET="BAT"                  # --dataset 인자
MODEL="TranAD"                            # --model 인자
NUM_EPOCHS=10                              # 고정 epoch

# 튜닝할 값들
WINDOW_LIST=(20)

FF_LIST=(16 32 64)

mkdir -p "$RESULTS_DIR"

RESULT_CSV="$RESULTS_DIR/test_auroc_summary.csv"
echo "model,n_window,dim_feedforward,val_auroc,log_path,config_path" > "$RESULT_CSV"

for w in "${WINDOW_LIST[@]}"; do
  for ld in "${FF_LIST[@]}"; do
    cfg_out="./configs/tranad_w${w}_fl${ld}.json"
    log_out="${RESULTS_DIR}/tranad_w${w}_fl${ld}.log"

    echo "==============================================="
    echo "Test: n_window=${w}, d_model_factor=${ld}"
    echo "Epochs: $NUM_EPOCHS"
    echo "Config: $cfg_out"
    echo "Log   : $log_out"
    echo "==============================================="


    # 2) main.py 실행 (VAL_AUROC를 로그에 찍게 되어 있어야 함)
    python3 main.py \
        --config "$cfg_out" \
        --dataset "$DATASET" \
	      --test
        --model "$MODEL" 2>&1 | tee "$log_out" \

    # 3) 로그에서 VAL_AUROC=... 라인 파싱
    val_auroc=$(grep "VAL_AUROC" "$log_out" | tail -n 1 | awk -F'=' '{print $2}')

    if [ -z "$val_auroc" ]; then
      val_auroc="NaN"
    fi

    echo "${MODEL},${w},${ld},${val_auroc},${log_out},${cfg_out}" >> "$RESULT_CSV"
  done
done

echo "==============================================="
echo "완료! 결과 요약: $RESULT_CSV"
cat "$RESULT_CSV"

