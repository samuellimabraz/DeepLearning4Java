package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.losses;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class CategoricalCrossEntropy implements ILossFunction {
    private final double eps = 1e-5f;
    @Override
    public INDArray forward(INDArray predicted, INDArray real) {
        INDArray logprobs = real.mul(Transforms.log(Util.clip(predicted, eps, 1 - eps))).negi();
//        System.out.println("predicted: " + predicted);
//        System.out.println("real: " + real);
//        System.out.println("logprobs: " + logprobs);
        return logprobs.sum(1).divi(real.columns());
    }

    @Override
    public INDArray backward(INDArray predicted, INDArray real) {
        return predicted.subi(real).divi(real.columns());
    }
}



