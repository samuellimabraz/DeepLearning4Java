package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Conv2D extends TrainableLayer {
    protected int filters;
    protected int kernelSize;
    protected int[] strides;
    protected String padding;
    protected IActivation activation;
    protected String kernelInitializer;
    protected int pad;

    protected ZeroPadding2D zeroPadding2D;
    protected int m, nh, nw, nc;

    private boolean isInitialized = false;

    public Conv2D(int filters, int kernelSize, int[] strides, String padding, IActivation activation, String kernelInitializer) {
        this.filters = filters;
        this.kernelSize = kernelSize;
        this.strides = strides;
        this.padding = padding;
        this.activation = activation;
        this.kernelInitializer = kernelInitializer;
    }

    public  Conv2D(int filters, int kernelSize, String padding, IActivation activation,  String kernelInitializer) {
        this(filters, kernelSize, new int[]{1, 1}, padding, activation, kernelInitializer);
    }

    public Conv2D(int filters, int kernelSize, IActivation activation) {
        this(filters, kernelSize, "valid", activation, "random");
    }

    public Conv2D(int filters, int kernelSize) {
        this(filters, kernelSize, "valid", Activation.create("linear"), "random");
    }

    @Override
    public void setup(INDArray inputs) {
        // Implement the setup method for a convolution function
        long[] shape = inputs.shape();
        this.m = (int) shape[0];
        int nHPrev = (int) shape[1], nWPrev = (int) shape[2];
        this.nc = (int) shape[3];

        if (padding.equals("same")) {
            this.pad = kernelSize / 2;
        } else if (padding.equals("valid")) {
            this.pad = 0;
        } else {
            throw new IllegalArgumentException("Invalid padding: " + padding);
        }
        this.zeroPadding2D = new ZeroPadding2D(pad);

        this.nh = ((nHPrev - kernelSize + 2 * pad) / strides[0] + 1);
        this.nw = ((nWPrev - kernelSize + 2 * pad) / strides[1] + 1);

        double scale = getScale();

        // Weights and bias represented as a single 5D tensor, where the last dimension is used for the bias
        // The tensor has shape (kernelSize, kernelSize, nc, filters, 2)
        params = Nd4j.create(DataType.DOUBLE, kernelSize, kernelSize, nc, filters, 2);

        INDArray W = Nd4j.randn(DataType.DOUBLE, kernelSize, kernelSize, nc, filters).mul(scale);
        INDArray b = Nd4j.randn(DataType.DOUBLE, kernelSize, kernelSize, nc, filters).mul(scale);

        params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(W);
        params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)).assign(b);

        // Gradients of weights and bias
        grads = Nd4j.zeros(DataType.DOUBLE, params.shape());

        this.isInitialized = true;
    }

    private double getScale() {
        double scale;
        if (this.kernelInitializer.equals("xavier")) {
            if (this.activation instanceof Sigmoid || this.activation instanceof LeakyReLU || this.activation instanceof Linear) {
                scale = Math.sqrt(6.0 / (nc + filters));
            } else if (this.activation instanceof ReLU || this.activation instanceof Softmax) {
                scale = Math.sqrt(2.0 / (nc + filters));
            } else if (this.activation instanceof TanH || this.activation instanceof SiLU) {
                scale = Math.sqrt(1.0 / nc); // Lecun initialization
            } else {
                throw new IllegalArgumentException("Xavier initialization not supported for this activation function");
            }
        } else if (this.kernelInitializer.equals("zero")) {
            scale = 0.0;
        } else {
            scale = 0.01;
        }
        return scale;
    }

    @Override
    public INDArray forward(INDArray inputs) {
        if (!this.isInitialized) {
            this.setup(inputs);
        }
        // Implement the forward propagation for a convolution function
        INDArray Z = Nd4j.zeros(m, nh, nw, filters);

        // Apply padding to the input volume
        INDArray paddedInputs = zeroPadding2D.forward(inputs);

        // Loop over the batch of training examples
        for (int i = 0; i < m; i++) {
            INDArray aPrev = paddedInputs.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            // Loop over vertical axis of the output volume
            for (int h = 0; h < nh; h++) {
                // Find the vertical start and end of the current "slice"
                int vert_start = h * strides[0];
                int vert_end = vert_start + kernelSize;

                // Loop over horizontal axis of the output volume
                for (int w = 0; w < nw; w++) {
                    // Find the horizontal start and end of the current "slice"
                    int horiz_start = w * strides[1];
                    int horiz_end = horiz_start + kernelSize;

                    // Loop over channels (= #filters) of the output volume
                    for (int c = 0; c < filters; c++) {
                        // Use the corners to define the (3D) slice of inputs
                        INDArray aSlicePrev = aPrev.get(NDArrayIndex.interval(vert_start, vert_end), NDArrayIndex.interval(horiz_start, horiz_end), NDArrayIndex.all());

                        // Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                        INDArray weights = params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.point(0));
                        INDArray biases = params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.point(1));

                        Z.putScalar(new int[]{i, h, w, c}, aSlicePrev.mul(weights).sumNumber().doubleValue() + biases.getDouble(0));
                    }
                }
            }
        }

        return activation.forward(Z);
    }

    @Override
    public INDArray backward(INDArray gradout) {
        // Implement the backward propagation for a convolution function
        return null;
    }
}
