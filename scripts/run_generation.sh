MODEL=$1
DATASET=$2
TASK=$3

# Check that the HF_HOME environment variable is set (if not, default shoulb be "~/.cache/huggingface")
if [ -z "$HF_HOME" ]
then
    echo "The enviroment variable HF_HOME must be specified in order to run this script!"
    return 1
fi

# Get docker image name (convert model name to lower case)
DOCKER_IMAGE=$(echo "$MODEL" | awk '{print tolower($0)}')
OUTPUT_FILE="GLUE3D/answers-$MODEL-$DATSET-$TASK.csv"

docker run --gpus all --rm -v $HF_HOME:/root/.cache/huggingface -v .:/GLUE3D \
  "$DOCKER_IMAGE" sh -c "PYTHONPATH=/GLUE3D:. python /GLUE3D/glue3d/main.py generate --task $TASK --model $MODEL --output-file \"$OUTPUT_FILE\" --data $DATASET"
