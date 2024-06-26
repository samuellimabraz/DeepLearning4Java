package br.deeplearning4java.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IMetric {
    double evaluate(INDArray yTrue, INDArray yPred);
}
