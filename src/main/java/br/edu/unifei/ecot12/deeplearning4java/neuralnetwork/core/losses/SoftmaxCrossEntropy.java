package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.losses;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.Softmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import static br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.Util.*;

public class SoftmaxCrossEntropy implements ILossFunction {
    private double eps = 1e-9;
    private boolean singleClass = false;

    private final Softmax softmax = new Softmax();

    public  SoftmaxCrossEntropy() {}

    public SoftmaxCrossEntropy(double eps) {
        this.eps = eps;
    }

    @Override
    public INDArray forward(INDArray predicted, INDArray real) {
        if (real.shape()[1] == 0) {
            singleClass = true;
            System.out.println("Single class");
        }

        if (singleClass) {
            predicted = normalize(predicted);
            real = normalize(real);
        }

        INDArray softmaxPreds = softmax.forward(predicted);
        softmaxPreds = clip(softmaxPreds, eps, 1 - eps);

        INDArray softmaxCrossEntropyLoss = real.mul(-1).mul(Transforms.log(softmaxPreds)).subi(
                real.rsub(1).mul(Transforms.log(softmaxPreds.rsub(1))));

        return softmaxCrossEntropyLoss.sum().div(predicted.shape()[0]);
    }

    @Override
    public INDArray backward(INDArray predicted, INDArray real) {
        INDArray softmaxPreds = softmax.forward(predicted);
        softmaxPreds = clip(softmaxPreds, eps, 1 - eps);

        if (singleClass) {
            return unnormalize(softmaxPreds.sub(real));
        } else {
            return softmaxPreds.sub(real).div(predicted.shape()[0]);
        }
    }
}
