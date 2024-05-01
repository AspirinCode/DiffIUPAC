#!/bin/bash

#!/bin/bash

MODEL_DIR=./ckpts/wjm_ckpts/wjm14_128_0.0001_2000_1000000_10000_schegran20000_srciupac_tgtsmiles
MODEL_NAME=${MODEL_DIR}/$1
OUT_DIR=${2}
SCHEDULE_PATH=${MODEL_DIR}/${3}
VAL_TXT=./data/${4}/seed
SEED=${5:-10708}

if [ -z "$OUT_DIR" ]; then
    OUT_DIR=${MODEL_NAME}
fi

GEN_BY_Q=${6:-"False"}
GEN_BY_MIX=${7:-"True"}
MIX_PROB=${8:-1}
MIX_PART=${9:-0}
TOP_P=-1
CLAMP="no_clamp"
BATCH_SIZE=50
SEQ_LEN=128
DIFFUSION_STEPS=2000
NUM_SAMPLES=-1



python -u inference_main_opt_input.py --model_name_or_path ${MODEL_NAME} --sequence_len_src 1024 \
--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --top_p ${TOP_P} --time_schedule_path ${SCHEDULE_PATH} \
--seed ${SEED} --val_txt_path ${VAL_TXT} --generate_by_q ${GEN_BY_Q} --generate_by_mix ${GEN_BY_MIX} \
--out_dir ${OUT_DIR} --diffusion_steps ${DIFFUSION_STEPS} --clamp ${CLAMP} --sequence_len ${SEQ_LEN} \
  --generate_by_mix_prob ${MIX_PROB} --generate_by_mix_part ${MIX_PART}




#bash ./inference_scrpts/non_translation_inf_opt.sh ema_0.9999_160000.pt ./output alpha_cumprod_step_160000.npy wjm14
#https://mp.weixin.qq.com/s/v8wHFwRm3IbGgnGD0syTgw