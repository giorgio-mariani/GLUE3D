MODEL=pointllm_7B
# Check that the HF_HOME environment variable is set (if not, default shoulb be "~/.cache/huggingface")
if [ -z "$HF_HOME" ]
then
    echo "The enviroment variable HF_HOME must be specified in order to run this script!"
    return 1
fi

# Get docker image name (convert model name to lower case)
DOCKER_IMAGE=$(echo "$MODEL" | awk '{print tolower($0)}')

docker run --gpus all --rm -it -v $HF_HOME:/root/.cache/huggingface -v .:/GLUE3D \
    -p 8888:8888 "$DOCKER_IMAGE" sh -c "pip install notebook;PYTHONPATH=/GLUE3D:/workdir jupyter notebook /GLUE3D/notebooks/example-pointllm.ipynb --ip 0.0.0.0 --no-browser --allow-root"
