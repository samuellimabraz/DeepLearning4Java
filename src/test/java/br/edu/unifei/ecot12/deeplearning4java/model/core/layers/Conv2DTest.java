package br.edu.unifei.ecot12.deeplearning4java.model.core.layers;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.Conv2D;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class Conv2DTest {

    @Test
    public void testForward() {
        // Create an instance of Conv2D
        int filters = 4;
        int kernelSize = 2;
        int[] strides = {1, 1};
        String padding = "same";
        Conv2D conv2D = new Conv2D(filters, kernelSize, strides, padding, Activation.create("relu"), "xavier");

        // Create some input data
        INDArray inputs = Nd4j.rand(1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        // Call the forward method
        INDArray output = conv2D.forward(inputs);

        System.out.println("Params shape:" + conv2D.getParams().shapeInfoToString());
        System.out.println("Params:" + conv2D.getParams());

        System.out.println("Weights shape:" + conv2D.getWeights().shapeInfoToString());
        System.out.println("Weights:" + conv2D.getWeights());
        System.out.println("Biases shape:" + conv2D.getBiases().shapeInfoToString());
        System.out.println("Biases:" + conv2D.getBiases());

        // Check the shape of the output
        long[] shape = output.shape();

        System.out.println("Output:" + output);
        System.out.println("Output Shape:" + output.shapeInfoToString());

        assertNotNull(shape);
        assertEquals(4, shape.length);
        assertEquals(1, shape[0]); // should match the batch size of the input
        assertEquals(6, shape[1]); // should be (input height - kernel size + 2 * pad) / stride + 1
        assertEquals(6, shape[2]); // should be (input width - kernel size + 2 * pad) / stride + 1
        assertEquals(filters, shape[3]); // should match the number of filters
    }

    @Test
    public void testBackward() {
        // Create an instance of Conv2D
        int filters = 4;
        int kernelSize = 2;
        int[] strides = {1, 1};
        String padding = "same";
        Conv2D conv2D = new Conv2D(filters, kernelSize, strides, padding, Activation.create("relu"), "xavier");

        // Create some input data
        INDArray inputs = Nd4j.rand(1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        // Call the forward method
        INDArray output = conv2D.forward(inputs);

        System.out.println("Output shape:" + output.shapeInfoToString());

        // Create some gradient data
        INDArray gradout = Nd4j.rand(output.shape());

        // Call the backward method
        INDArray gradInput = conv2D.backward(gradout);

        // Check the shape of the gradient input
        long[] shape = gradInput.shape();

        System.out.println("GradInput shape:" + gradInput.shapeInfoToString());

        assertNotNull(shape);
        assertEquals(4, shape.length);
        assertEquals(1, shape[0]); // should match the batch size of the input
        assertEquals(5, shape[1]); // should match the height of the input
        assertEquals(5, shape[2]); // should match the width of the input
        assertEquals(3, shape[3]); // should match the number of channels of the input
    }
}
