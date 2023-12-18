package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Dropout extends Layer<Dropout> {
    private double dropoutRate = 0.0;
    private INDArray mask;

    public Dropout(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }

    public Dropout() {

    }

    @Override
    public void setup(INDArray inputs) {
        // No setup needed for dropout layer
    }

    @Override
    public INDArray forward(INDArray inputs) {
        
        if (dropoutRate > 0) {
            this.mask = Nd4j.rand(inputs.shape()).gt(dropoutRate);
            this.setOutput(inputs.mul(this.mask));
        } else {
            this.setOutput(inputs);
        }

        return this.getOutput();
    }

    @Override
    public INDArray backward(INDArray gradout) {
        if (dropoutRate > 0) {
            return gradout.mul(this.mask);
        } else {
            return gradout;
        }
    }

    @Override
    public String toString() {
        return "Dropout(" + this.dropoutRate + ")";
    }

    public double getDropoutRate() {
        return this.dropoutRate;
    }

    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }

    public INDArray getMask() {
        return this.mask;
    }

    public void setMask(INDArray mask) {
        this.mask = mask;
    }

    /**
     * @param dos
     * @throws IOException
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        dos.writeDouble(this.dropoutRate);
    }


    /**
     * @param dis
     * @return
     * @throws IOException
     */
    @Override
    public Dropout loadAdditional(DataInputStream dis) throws IOException {
        double dropoutRate = dis.readDouble();
        return new Dropout(dropoutRate);
    }

}
