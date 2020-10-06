import numpy as np
import cv2
import onnx
import onnxruntime
from argparse import ArgumentParser

    
def inference(onnx_filename,image_filename):
    
    sess = onnxruntime.InferenceSession(onnx_filename)
    sess.set_providers(['CPUExecutionProvider'])
    
    # input name and shape
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # output name
    output_name = sess.get_outputs()[0].name  
    
    
    #----------------------load image----------------------#
    img = cv2.imread(image_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224),interpolation = cv2.INTER_NEAREST)
    img = img.reshape(-1,224,224,3)
    img = img.astype(np.float32)
    #----------------------load image----------------------#
    
    
    outputs = sess.run([output_name], {input_name: img})[0]
    outputs = np.argmax(outputs)
    # outputs = softmax(np.array(outputs)) # not necessary
    
    print("predicted class = ",outputs)
    print('finifhsed\n\n')
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-o", "--onnx_filename", help = "filename of converted model")
    parser.add_argument("-i", "--image_filename",  help = "filename of converted model")
    args = parser.parse_args()
    if args.onnx_filename and args.image_filename:
        inference(args.onnx_filename, args.image_filename)
        
        