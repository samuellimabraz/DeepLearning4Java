package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Flatten extends Layer<Flatten> {
    @Override
    public void setup(INDArray inputs) {
        // Não há parâmetros para configurar na camada Flatten
    }

    @Override
    public INDArray forward(INDArray inputs) {
        this.setInput(inputs.dup());
        INDArray output = inputs.reshape(new int[] {(int) inputs.size(0), -1 });
        this.setOutput(output);
        return output;
    }

    @Override
    public INDArray backward(INDArray gradout) {
        INDArray gradInput = gradout.reshape(this.getInput().shape());
        return gradInput;
    }

    @Override
    public String toString() {
        return "Flatten";
    }

    /**
     * @param dis
     * @return
     * @throws IOException
     */
    @Override
    public Flatten loadAdditional(DataInputStream dis) throws IOException {
        return null;
    }

    /**
     * @param dos
     * @throws IOException
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {

    }
}
