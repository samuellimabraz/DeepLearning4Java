package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Layer <T extends Layer<T>> {
    protected INDArray input;
    protected INDArray output;

    public void setup(INDArray inputs) {
        this.input = inputs;
    }

    public abstract INDArray forward(INDArray inputs);
    public abstract INDArray backward(INDArray gradout);

    public String toString() {
        return this.getClass().getSimpleName();
    }

    public T load(DataInputStream dis) throws IOException {
        input = Nd4j.read(dis);
        output = Nd4j.read(dis);
        return loadAdditional(dis);
    }

    public abstract T loadAdditional(DataInputStream dis) throws IOException;

    public void save(DataOutputStream dos) throws IOException {
        dos.writeUTF(this.getClass().getSimpleName());
        if (input == null || output == null) {
            throw new IOException("Layer is not initialized");
        }
        Nd4j.write(input, dos);
        Nd4j.write(output, dos);
        this.saveAdditional(dos);
    }
    public abstract void saveAdditional(DataOutputStream dos) throws IOException;

    public INDArray getInput() {
        return input;
    }
    public INDArray getOutput() {
        return output;
    }

    public void setInput(INDArray input) {
        this.input = input;
    }
    public void setOutput(INDArray output) {
        this.output = output;
    }
}
