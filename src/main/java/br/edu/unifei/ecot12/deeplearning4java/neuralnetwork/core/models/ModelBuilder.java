package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class ModelBuilder {
    public List<Layer> layers = new ArrayList<>();

    public ModelBuilder add(Layer layer) {
        this.layers.add(layer);
        return this;
    }
    public NeuralNetwork build() {
        return new NeuralNetwork(this);
    }

}

