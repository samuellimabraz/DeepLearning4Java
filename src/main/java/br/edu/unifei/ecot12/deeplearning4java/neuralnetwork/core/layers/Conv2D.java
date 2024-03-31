package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
    protected int m ,nHInput, nWInput, nCInput;

    protected int nHOutput, nWOutput, nCOutput;

    private boolean isInitialized = false;

    private INDArray paddedInputs;

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
        this.nHInput = (int) shape[1];
        this.nWInput = (int) shape[2];
        this.nCInput = (int) shape[3];

        if (padding.equals("same")) {
            this.pad = kernelSize / 2;
        } else if (padding.equals("valid")) {
            this.pad = 0;
        } else {
            throw new IllegalArgumentException("Invalid padding: " + padding);
        }
        this.zeroPadding2D = new ZeroPadding2D(pad);

        this.nHOutput = ((nHInput- kernelSize + 2 * pad) / strides[0] + 1);
        this.nWOutput = ((nWInput - kernelSize + 2 * pad) / strides[1] + 1);
        this.nCOutput = filters;

        double scale = getScale();

        // Weights and bias represented as a single 5D tensor, where the last dimension is used for the bias
        // The tensor has shape (kernelSize, kernelSize, nc, filters, 2)
        params = Nd4j.create(DataType.DOUBLE, kernelSize, kernelSize, nCInput, filters, 2);

        INDArray W = Nd4j.randn(DataType.DOUBLE, kernelSize, kernelSize, nCInput, filters).mul(scale);
        INDArray b = Nd4j.randn(DataType.DOUBLE, 1, 1, 1, filters).mul(scale); // Initialize b with shape (1, 1, 1, filters)

        // Broadcast b to have the same shape as W
        b = b.broadcast(kernelSize, kernelSize, nCInput, filters);

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
                scale = Math.sqrt(6.0 / (nCInput + filters));
            } else if (this.activation instanceof ReLU || this.activation instanceof Softmax) {
                scale = Math.sqrt(2.0 / (nCInput + filters));
            } else if (this.activation instanceof TanH || this.activation instanceof SiLU) {
                scale = Math.sqrt(1.0 / nCInput); // Lecun initialization
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
        setInput(inputs);

        this.m = (int) inputs.shape()[0];
        // Implement the forward propagation for a convolution function
        output = Nd4j.zeros(m, nHOutput, nWOutput, filters);

        // Apply padding to the input volume
        this.paddedInputs = zeroPadding2D.forward(inputs);

        // Loop over the batch of training examples
        for (int i = 0; i < m; i++) {
            INDArray aPrev = paddedInputs.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            // Loop over vertical axis of the output volume
            for (int h = 0; h < nHInput; h++) {
                // Find the vertical start and end of the current "slice"
                int vert_start = h * strides[0];
                int vert_end = vert_start + kernelSize;

                // Loop over horizontal axis of the output volume
                for (int w = 0; w < nWOutput; w++) {
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

                        output.putScalar(new int[]{i, h, w, c}, aSlicePrev.mul(weights).sumNumber().doubleValue() + biases.getDouble(0));
                    }
                }
            }
        }

        output = activation.forward(output);

        return output;
    }

    @Override
    public INDArray backward(INDArray gradout) {
        // Initialize dA_prev, dW, db with the correct shapes
        long[] dZshape = gradout.shape();
        long m = dZshape[0];
        long nH = dZshape[1];
        long nW = dZshape[2];
        long nC = dZshape[3];

        INDArray dA_prev = Nd4j.zeros(m, nHInput, nWInput, nCInput);
        INDArray dW = Nd4j.zeros(getWeights().shape());
        INDArray db = Nd4j.zeros(getBiases().shape());

        System.out.println("dA_prev shape:" + dA_prev.shapeInfoToString());
        System.out.println("dW shape:" + dW.shapeInfoToString());
        System.out.println("db shape:" + db.shapeInfoToString());

        // Pad inputs and dA_prev
        INDArray dA_prev_pad = zeroPadding2D.forward(dA_prev);

        for (int i = 0; i < m; i++) {
            INDArray a_prev_pad = paddedInputs.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray da_prev_pad = dA_prev_pad.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

            for (int h = 0; h < nH; h++) {
                int vert_start = h * strides[0];
                int vert_end = vert_start + kernelSize;

                for (int w = 0; w < nW; w++) {
                    int horiz_start = w * strides[1];
                    int horiz_end = horiz_start + kernelSize;

                    for (int c = 0; c < filters; c++) {
                        INDArray a_slice = a_prev_pad.get(NDArrayIndex.interval(vert_start, vert_end), NDArrayIndex.interval(horiz_start, horiz_end), NDArrayIndex.all());

                        INDArray weights = params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.point(0));
                        //INDArray biases = params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.point(1));

                        INDArray dZ_slice = gradout.get(NDArrayIndex.point(i), NDArrayIndex.point(h), NDArrayIndex.point(w), NDArrayIndex.point(c));

                        da_prev_pad.get(NDArrayIndex.interval(vert_start, vert_end), NDArrayIndex.interval(horiz_start, horiz_end), NDArrayIndex.all()).addi(weights.mul(dZ_slice));
                        dW.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c)).addi(a_slice.mul(dZ_slice));
                        db.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c)).addi(dZ_slice);
                    }
                }
            }

            int endIdxH = (int) (dA_prev.shape()[1] - pad);
            int endIdxW = (int) (dA_prev.shape()[2] - pad);
            dA_prev.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(pad, endIdxH), NDArrayIndex.interval(pad, endIdxW), NDArrayIndex.all()},
                    da_prev_pad.get(NDArrayIndex.interval(pad, endIdxH), NDArrayIndex.interval(pad, endIdxW), NDArrayIndex.all()));
        }

        // Update gradients in params
        grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(dW);
        grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)).assign(db);

        return dA_prev;
    }

    public INDArray getWeights() {
        return params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
    }

    public INDArray getBiases() {
        return params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1));
    }

    public INDArray getGradWeights() {
        return grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
    }

    public INDArray getGradBiases() {
        return grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1));
    }
}
