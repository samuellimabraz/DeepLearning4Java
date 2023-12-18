package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IActivation {
    INDArray forward(INDArray input);
    INDArray backward(INDArray input);
}
