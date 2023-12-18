package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

public class Dense extends TrainableLayer {
    private IActivation activation;
    private int units;
    private boolean isInitialized = false;
    private String kernelInitializer;

    private double lambda;

    /**
     * Constructor for Dense
     *
     * @param units number of neurons in the layer. (int)
     * @param activation activation function of the layer. (IActivation)
     * @param kernelInitializer weight initializer of the layer. (String)
     *
     */
    public Dense(int units, IActivation activation, String kernelInitializer, double lambda) {
        this.units = units;
        this.activation = activation;
        this.kernelInitializer = kernelInitializer;
        this.lambda = lambda;
    }

    /**
     * Constructor for Dense
     *
     * @param units number of neurons in the layer. (int)
     * @param activation activation function of the layer. (IActivation)
     *
     */
    public Dense(int units, IActivation activation, String kernelInitializer) {
        this(units, activation, kernelInitializer, 0.01);
    }

    /**
     * Constructor for Dense
     *
     * @param units number of neurons in the layer. (int)
     * @param activation activation function of the layer. (IActivation)
     *
     */
    public Dense(int units, IActivation activation) {
        this(units, activation, "standard", 0.01);
    }

    /**
     * Constructor for Dense
     *
     * @param units number of neurons in the layer. (int)
     *
     */
    public Dense(int units) {
        this(units, Activation.create("relu"), "standard", 0.01);
    }

    public Dense() {
        this(0, Activation.create("relu"), "standard", 0.01);
    }

    public INDArray getWeights() {
        return this.params.get(NDArrayIndex.interval(0, this.params.rows() - 1), NDArrayIndex.all());
    }

    public INDArray getGradientWeights() {
        return this.grads.get(NDArrayIndex.interval(0, this.grads.rows() - 1), NDArrayIndex.all());
    }

    public INDArray getBias() {
        return this.params.getRow(this.params.rows() - 1);
    }

    public INDArray getGradientBias() {
        return this.grads.getRow(this.grads.rows() - 1);
    }

    /**
     * Initializes the weights and bias of the layer.
     *
     * @param inputs input of the layer. (INDArray)
     */
    @Override
    public void setup(INDArray inputs) {
        int numInputs = inputs.columns();
        double scale;
        if (this.kernelInitializer.equals("xavier")) {
            if (this.activation instanceof Sigmoid || this.activation instanceof LeakyReLU || this.activation instanceof Linear) {
                scale = Math.sqrt(6.0 / (numInputs + this.units));
            } else if (this.activation instanceof ReLU || this.activation instanceof Softmax) {
                scale = Math.sqrt(2.0 / (numInputs + this.units));
            } else if (this.activation instanceof TanH || this.activation instanceof SiLU) {
                scale = Math.sqrt(1.0 / numInputs); // Lecun initialization
            } else {
                throw new IllegalArgumentException("Xavier initialization not supported for this activation function");
            }
        } else if (this.kernelInitializer.equals("zero")) {
            scale = 0.0;
        } else {
            scale = 0.01;
        }
        // Weights and bias represented as a single matrix, where the last row is the bias
        // The matrix have shape (numInputs + 1, units)
        params = Nd4j.vstack(Nd4j.randn(DataType.DOUBLE, numInputs, this.units).mul(scale), Nd4j.randn(DataType.DOUBLE, 1, this.units).mul(scale)); // W and b
        // Gradients of weights and bias
        grads = Nd4j.zeros(DataType.DOUBLE, params.shape()); // dW and db
        isInitialized = true;
    }

    /**
     * Calculates the output of the layer.
     *
     * @param inputs input of the layer. (INDArray)
     *
     * @return output of the layer. (INDArray)
     */
    @Override
    public INDArray forward(INDArray inputs) {
        if (!this.isInitialized) {
            this.setup(inputs);
        }
        if (inputs.columns() != params.rows() - 1) {
            System.out.println("Inputs: " + inputs.columns());
            System.out.println("Params: " + params.rows());
            throw new IllegalArgumentException("Input dimensions are not compatible with weight matrix dimensions");
        }

         input = inputs;

        // output = activation(input * W + b)
        output = activation.forward(
                inputs.mmul(this.getWeights())
                .addi(this.getBias())
        );

        return output;
    }

    /**
     * Calculates the gradient of the layer.
     *
     * @param gradout gradient of the output of the layer. (INDArray)
     *
     * @return gradient of the input of the layer. (INDArray)
     */
    @Override
    public INDArray backward(INDArray gradout) {
        if (gradout.columns() != this.units) {
            throw new IllegalArgumentException("Gradient dimensions are not compatible with the number of units in the layer");
        }
        // Calculate gradient of output
        // dL/dx = dL/dy * dy/dx = dL/dy * activation'(input * W + b)
        INDArray gradInput = gradout.mul(this.activation.backward(output));

        // Calculate gradients of weights and bias
        grads = Nd4j.vstack(
                getInput().transpose().mmul(gradInput).addi(this.getWeights().mul(lambda)), // dW = input.T * gradInput
                gradInput.sum(0).reshape(1, units) // db = sum(gradInput)
        );

        // Calculate gradient of input
        return gradInput.mmul(this.getWeights().transpose()); // gradInput * W.T
    }

    /**
     * Returns the activation function of the layer.
     *
     * @return activation function of the layer. (IActivation)
     */
    public IActivation getActivation() {
        return activation;
    }

    /**
     * Returns the number of neurons in the layer.
     *
     * @return number of neurons in the layer. (int)
     */
    public int getUnits() {
        return units;
    }

    /**
     * Returns the weight initializer of the layer.
     *
     * @return weight initializer of the layer. (String)
     */
    public String getKernelInitializer() {
        return kernelInitializer;
    }

    /**
     * Returns the string representation of the layer.
     *
     * @return string representation of the layer. (String)
     */
    @Override
    public String toString() {
        String sb = "Dense(units=" + units +
                ", kernelInitializer=" + kernelInitializer +
                ", activation=" + activation.getClass().getSimpleName() +
                ", params=(" + "W=" + Arrays.toString(getWeights().reshape(1, getWeights().length()).toFloatVector()) + ", b=" + Arrays.toString(getBias().toFloatVector()) +
                "), grads=(" + "dW=" + Arrays.toString(getGradientWeights().reshape(1, getGradientWeights().length()).toFloatVector()) + ", db=" + Arrays.toString(getGradientBias().toFloatVector()) +
                "))";
        return sb;
    }

    /**
     * Saves the layer to a file.
     *  Writes the layer to a file using the following format:
     *  <ul>
     *      <li>Layer class name (String)</li>
     *      <li>Input of the layer (INDArray)</li>
     *      <li>Output of the layer (INDArray)</li>
     *      <li>Number of neurons in the layer (int)</li>
     *      <li>Activation function of the layer (String)</li>
     *      <li>Weight initializer of the layer (String)</li>
     *      <li>Is the layer initialized? (boolean)</li>
     *      <li>Weights of the layer (INDArray)</li>
     *      <li>Bias of the layer (INDArray)</li>
     *      <li>Gradients of the weights of the layer (INDArray)</li>
     *      <li>Gradients of the bias of the layer (INDArray)</li>
     * @param dos output stream. (DataOutputStream)
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        dos.writeInt(this.units);
        dos.writeUTF(this.activation.getClass().getSimpleName().toLowerCase());
        dos.writeUTF(this.kernelInitializer);
        dos.writeBoolean(this.isInitialized);
        super.saveAdditional(dos);
    }

    /**
     * Loads the layer from a file.
     * <p>
     * Reads the layer from a file using the following format:
     * <ul>
     *      <li>Layer class name (String)</li>
     *      <li>Input of the layer (INDArray)</li>
     *      <li>Output of the layer (INDArray)</li>
     *      <li>Number of neurons in the layer (int)</li>
     *      <li>Activation function of the layer (String)</li>
     *      <li>Weight initializer of the layer (String)</li>
     *      <li>Is the layer initialized? (boolean)</li>
     *      <li>Weights of the layer (INDArray)</li>
     *      <li>Bias of the layer (INDArray)</li>
     *      <li>Gradients of the weights of the layer (INDArray)</li>
     *      <li>Gradients of the bias of the layer (INDArray)</li>
     * @param dis input stream. (DataInputStream)
     * @return loaded layer. (Dense)
     */
    public Dense loadAdditional(DataInputStream dis) throws IOException {
        this.units = dis.readInt();
        this.activation = Activation.create(dis.readUTF());
        this.kernelInitializer = dis.readUTF();
        this.isInitialized = dis.readBoolean();
        super.loadAdditional(dis);
        return this;
    }

    public Dense(Dense dense) {
        this.units = dense.units;
        this.activation = dense.activation;
        this.kernelInitializer = dense.kernelInitializer;
        this.params = dense.params;
        this.grads = dense.grads;
        this.isInitialized = dense.isInitialized;
    }

}