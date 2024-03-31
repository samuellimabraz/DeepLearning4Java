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

        System.out.println("Params:" + conv2D.getParams());

        // Check the shape of the output
        long[] shape = output.shape();

        System.out.println("Output:" + output);
        System.out.println("Shape:" + output.shapeInfoToString());

        assertNotNull(shape);
        assertEquals(4, shape.length);
        assertEquals(1, shape[0]); // should match the batch size of the input
        assertEquals(6, shape[1]); // should be (input height - kernel size + 2 * pad) / stride + 1
        assertEquals(6, shape[2]); // should be (input width - kernel size + 2 * pad) / stride + 1
        assertEquals(filters, shape[3]); // should match the number of filters

        // Add more assertions to check the values of the output if necessary
    }
}
