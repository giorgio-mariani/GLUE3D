MODEL=$1
DATASET=$2
TASK=$3

# Check that the HF_HOME environment variable is set (if not, default shoulb be "~/.cache/huggingface")
if [ -z "$HF_HOME" ]
then
    echo "The enviroment variable HF_HOME must be specified in order to run this script!"
    exit 1
fi

# Answer Generation
OUTPUT_DIR="./results/$MODEL/$DATASET"
mkdir -p "$OUTPUT_DIR"

# Get docker image name (convert model name to lower case)
DOCKER_IMAGE=$(echo "$MODEL" | awk '{print tolower($0)}')
TARGET_FILE="$OUTPUT_DIR/$TASK.csv"

# Check if output file exists already
if [ -e "$TARGET_FILE" ]
then
    echo "$TARGET_FILE exists, skipping model MODEL=$MODEL and TASK=$TASK!"
    exit 0
else
    echo "$TARGET_FILE does not exists, starting evaluation for MODEL=$MODEL, DATASET=$DATASET, and TASK=$TASK!"
fi

docker run --gpus all --rm -v $HF_HOME:/root/.cache/huggingface -v .:/GLUE3D \
  "$DOCKER_IMAGE" sh -c "PYTHONPATH=/GLUE3D:. python /GLUE3D/glue3d/main.py generate --task $TASK --model $MODEL --output-file \"/GLUE3D/$TARGET_FILE\" --data $DATASET"
