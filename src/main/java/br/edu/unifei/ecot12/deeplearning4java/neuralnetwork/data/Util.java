package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Util {

    /**
     * Normalize the data (Dup the array)
     *
     * @param array
     * @return normalized array -> xi / sum(x)
     */
    public static INDArray normalize(INDArray array) {
        INDArray normalized = array.dup();
        normalized.divi(normalized.sumNumber());
        return normalized;
    }

    /**
     * Unnormalize the data (Dup the array)
     *
     * @param array
     * @return unnormalized array -> xi * sum(x)
     */
    public static INDArray unnormalize(INDArray array) {
        INDArray unnormalized = array.dup();
        unnormalized.muli(unnormalized.sumNumber());
        return unnormalized;
    }

    /**
     * Clip values between min and max (In-place)
     *
     * @param array
     * @param min
     * @param max
     * @return
     */
    public static INDArray clip(INDArray array, double min, double max) {
        Transforms.max(array, min, false);
        Transforms.min(array, max, false);
        return array;
    }

    /**
     * Split the data into train and test sets
     *
     * @param x
     * @param y
     * @param trainSize
     * @return [xTrain, yTrain], [xTest, yTest] (INDArray[][])
     */
    public static INDArray[][] trainTestSplit(INDArray x, INDArray y, double trainSize) {
        int numTrain = (int) (x.size(0) * trainSize);
        if (numTrain == x.size(0)) {
            return new INDArray[][] { { x, y }, { x, y } };
        }
        INDArray xTrain = x.get(NDArrayIndex.interval(0, numTrain));
        INDArray xTest = x.get(NDArrayIndex.interval(numTrain, x.size(0)));
        INDArray yTrain = y.get(NDArrayIndex.interval(0, numTrain));
        INDArray yTest = y.get(NDArrayIndex.interval(numTrain, y.size(0)));
        return new INDArray[][] { { xTrain, yTrain }, { xTest, yTest } };
    }

    /**
     * Print a progress bar
     *
     * @param current
     * @param total
     */
    public static void printProgressBar(int current, int total) {
        int percent = (int) ((current / (double) total) * 100);
        int progressBars = percent / 2;

        StringBuilder progressBar = new StringBuilder(50);
        progressBar.append("\r[");

        for (int i = 0; i < 50; i++) {
            if (i < progressBars) {
                progressBar.append("=");
            } else if (i == progressBars) {
                progressBar.append(">");
            } else {
                progressBar.append(" ");
            }
        }

        progressBar.append("] ");
        progressBar.append(percent);
        progressBar.append("%");

        System.out.print(progressBar);
    }

    /**
     * One-hot encode the labels
     *
     * @param labels
     * @param numClasses
     * @return labels one-hot encoded
     *         ex: [0, 1, 2] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
     */
    public static INDArray oneHotEncode(INDArray labels, int numClasses) {
        INDArray oneHotEncoded = Nd4j.zeros(DataType.DOUBLE, labels.rows(), numClasses);

        for (int i = 0; i < labels.rows(); i++) {
            int classIdx = labels.getInt(i);
            oneHotEncoded.putScalar(new int[]{i, classIdx}, 1);
        }

        return oneHotEncoded;
    }

}
