package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class ZeroPadding2D extends Layer<ZeroPadding2D> {
    private int padding;

    public ZeroPadding2D(int padding) {
        this.padding = padding;
    }

    @Override
    public INDArray forward(INDArray inputs) {
        this.setInput(inputs.dup());
        INDArray output = Nd4j.pad(inputs, new int[][]{{0, 0}, {padding, padding}, {padding, padding}, {0, 0}});
        this.setOutput(output);
        return output;
    }

    @Override
    public INDArray backward(INDArray gradout) {
        INDArray gradInput = gradout.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(padding, gradout.shape()[1] - padding),
                NDArrayIndex.interval(padding, gradout.shape()[2] - padding),
                NDArrayIndex.all()
        );
        return gradInput;
    }

    @Override
    public ZeroPadding2D loadAdditional(DataInputStream dis) throws IOException {
        return null;
    }

    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        dos.writeInt(padding);
    }

    @Override
    public String toString(){
        return "ZeroPadding2D(padding=" + this.padding + ")";
    }
}
