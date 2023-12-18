package br.edu.unifei.ecot12.deeplearning4java.model.data;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.Util;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class UtilTest {

    @Test
    public void testNormalize() {
        Util util = new Util();
        INDArray x = Nd4j.create(new double[]{5, 3, 2});

        System.out.println(x);
    }
}
