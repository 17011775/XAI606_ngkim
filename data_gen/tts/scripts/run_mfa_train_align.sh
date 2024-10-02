#!/bin/bash
set -e

NUM_JOB=${NUM_JOB:-20}
echo "| Training MFA using ${NUM_JOB} cores."
BASE_DIR=/workspace/ng/data/processed/$CORPUS
echo "BASE_DIR: $BASE_DIR"

# Ensure the necessary directories and files exist
if [ ! -d "$BASE_DIR/mfa_inputs" ]; then
    echo "Error: Directory $BASE_DIR/mfa_inputs does not exist."
    exit 1
fi

if [ ! -f "$BASE_DIR/mfa_dict.txt" ]; then
    echo "Error: File $BASE_DIR/mfa_dict.txt does not exist."
    exit 1
fi

rm -rf $BASE_DIR/mfa_outputs_tmp

echo "Running MFA training..."
for wav_file in $BASE_DIR/mfa_inputs/*.wav; do
    echo "Processing file: $wav_file"
done

mfa train $BASE_DIR/mfa_inputs $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/mfa_model.zip --clean -j $NUM_JOB

if [ $? -ne 0 ]; then
    echo "Error: MFA training failed."
    exit 1
fi

rm -rf $BASE_DIR/mfa_tmp $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_outputs_tmp -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
rm -rf $BASE_DIR/mfa_outputs_tmp

echo "MFA training completed successfully."