import tensorflow as tf
import numpy as np

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        # Initializing the interpreter with the specified model and threads
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        # Allocating tensors for the interpreter
        self.interpreter.allocate_tensors()

        # Retrieving input and output details of the model
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Getting the tensor index for the input details
        input_details_tensor_index = self.input_details[0]['index']

        # Setting the input tensor with the provided landmark list
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))

        # Invoking the interpreter to perform inference
        self.interpreter.invoke()

        # Getting the tensor index for the output details
        output_details_tensor_index = self.output_details[0]['index']

        # Getting the result from the interpreter output tensor
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Finding the index with the highest confidence score
        result_index = np.argmax(np.squeeze(result))

        return result_index
