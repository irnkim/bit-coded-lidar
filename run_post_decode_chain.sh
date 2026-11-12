#!/bin/bash
# For every B already swept (bits saved from the earlier G=4x4/F=50 run),
# compute recon-cum-bilat (CUM_BILATERAL) and ICM-recon/ICM-bilat/
# ICM-cum-bilat (ICM_SWEEPS), reusing the saved bits via LOAD_BITS=1 --
# no resimulation. Skips a stage if its output already exists.
set -e
cd /lustre/nvwulf/projects/DaveGroup-nvwulf/irnkim/spad_sim/code_clean
PY=/lustre/nvwulf/software/miniconda3/envs/cupy/bin/python3
RESULTS=data/rendered_images/bathroom-cycles-2/results

export SCENE=bathroom-cycles-2
export G_H=4 G_W=4
export CODE=trig
export TRIG_K=50
export P_DETECT=0.3
export HW_DELAY_PROBS="0:0.25,1:0.25,2:0.25,3:0.25"
export LOAD_BITS=1
unset OUT_TAG CYCLES VERIFY_BATCHES SAVE_BITS

STEM_BASE="G4x4_B%s_shift0_trig_unshared_p0.3_majority0_trigF50_hwdelay0x0.25-1x0.25-2x0.25-3x0.25"

for B in 50 100 250 500 1000; do
    export B
    STEM=$(printf "$STEM_BASE" "$B")
    CUMB_OUT="$RESULTS/${STEM}_cumulative_vs_independent.npz"
    ICM_OUT="$RESULTS/${STEM}_iterative_decode.npz"

    if [ -f "$CUMB_OUT" ]; then
        echo "=== B=$B CUM_BILATERAL already done, skipping $(date) ==="
    else
        echo "=== B=$B CUM_BILATERAL $(date) ==="
        CUM_BILATERAL=1 ICM_SWEEPS=0 $PY run_nonadaptive.py
    fi

    if [ -f "$ICM_OUT" ]; then
        echo "=== B=$B ICM_SWEEPS already done, skipping $(date) ==="
    else
        echo "=== B=$B ICM_SWEEPS $(date) ==="
        CUM_BILATERAL=0 ICM_SWEEPS=5 $PY run_nonadaptive.py
    fi
done

echo "=== post-decode chain complete $(date) ==="
