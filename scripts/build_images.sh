DOCKER_ROOT="$(realpath $(dirname $0))/../docker_images" # <- get docker images path

docker build $DOCKER_ROOT/shapellm -t shapellm_7b
docker build $DOCKER_ROOT/pointllm -t pointllm_7b -t pointllm_13b
docker build $DOCKER_ROOT/minigpt3d -t minigpt3d
docker build $DOCKER_ROOT/llava3d -t llava3d
docker build $DOCKER_ROOT/phi3_vision -t phi_vision
docker build $DOCKER_ROOT/others -t qwen_vl -t llava -t llama_3 -t phi_3.5_mini -t vicuna_v1.1_7b -t vicuna_v1.1_13b -t llama_mesh
