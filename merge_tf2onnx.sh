set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
# conda activate TensorFlow2

########################################Step1############################################
################################export efficientnet model################################
#########################################################################################
MODEL_NAME="efficientnet-b0"
CKPT_DIR="./efficientnet-b0-baseline"
OUTPUT_TFLITE="./pretrained-"$MODEL_NAME".tflite"
QUANTIZE=false
OUTPUT_SAVED_MODEL_DIR="pretrained-"$MODEL_NAME
NUM_CLASSES=1000

rm -r -f $OUTPUT_SAVED_MODEL_DIR
mkdir $OUTPUT_SAVED_MODEL_DIR

python ./export_model.py \
--model_name=$MODEL_NAME \
--ckpt_dir=$CKPT_DIR \
--output_tflite=$OUTPUT_TFLITE \
--quantize=$QUANTIZE \
--output_saved_model_dir=$OUTPUT_SAVED_MODEL_DIR \

##########################################################################################


#########################################Step2###########################################
############################convert efficientnet model to onnx###########################
#########################################################################################
TENSORFLOW_MODEL_PATH=$OUTPUT_SAVED_MODEL_DIR

python -m tf2onnx.convert --saved-model=$TENSORFLOW_MODEL_PATH \
						--output=model.onnx

#########################################################################################


##########################################step3###########################################
################################do inference by the script################################
ONNX_FILENAME="model.onnx"
IMAGE_FILENAME="test_image.jpg"
python inference_tfonnx.py --onnx_filename=$ONNX_FILENAME \
						--image_filename=$IMAGE_FILENAME \

##########################################################################################


					