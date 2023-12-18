package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.optimizers;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.TrainableLayer;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class Optimizer {
    protected NeuralNetwork neuralNetwork;
    protected Map<TrainableLayer, List<INDArray>> auxParams = new HashMap<>();

    private boolean initialized = false;

    public Optimizer() {}

    public Optimizer(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    protected void init() {
        for (TrainableLayer layer : neuralNetwork) {
            auxParams.put(layer, createAuxParams(layer.getParams()));
        }
    }

    public void update() {
        if (!initialized) {
            init();
            initialized = true;
        }
        for (TrainableLayer layer : neuralNetwork) {
            updateRule(layer.getParams(), layer.getGrads(), auxParams.get(layer));
        }
    }

    protected abstract List<INDArray> createAuxParams(INDArray params);

    protected abstract void updateRule(INDArray params, INDArray grads, List<INDArray> auxParams);
}