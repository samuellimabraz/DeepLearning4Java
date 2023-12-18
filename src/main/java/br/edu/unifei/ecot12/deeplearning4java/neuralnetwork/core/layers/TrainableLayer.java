package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public abstract class TrainableLayer extends Layer<TrainableLayer> {
    protected INDArray params;
    protected INDArray grads;

    public INDArray getParams() { return params; }

    public void setParams(INDArray params) { this.params = params; }

    public INDArray getGrads() { return grads; }

    public void setGrads(INDArray grads) { this.grads = grads; }

    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        Nd4j.write((params == null) ? Nd4j.zeros(1) : params , dos);
        Nd4j.write((grads == null) ? Nd4j.zeros(1): grads, dos);
    }

    @Override
    public TrainableLayer loadAdditional(DataInputStream dis) throws IOException {
        params = Nd4j.read(dis);
        grads = Nd4j.read(dis);
        return this;
    }
}
