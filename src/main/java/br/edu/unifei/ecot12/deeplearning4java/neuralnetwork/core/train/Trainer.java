package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.losses.ILossFunction;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.Util;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.optimizers.Optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Trainer {
    private final NeuralNetwork model;
    private final Optimizer optimizer;
    private final ILossFunction lossFunction;
    
    private INDArray trainInputs;
    private INDArray trainTargets;
    private INDArray testInputs;
    private INDArray testTargets;

    private INDArray[] batch = new INDArray[2];

    private final int epochs;
    private final int batchSize;
    private int currentIndex;
    private final int patience;

    private int evalEvery = 10;
    private final boolean earlyStopping, verbose;

    private double bestLoss = Double.MAX_VALUE;

    private int wait = 0;

    private final double threshold = Double.MIN_VALUE;

    private double valLoss;

    public Trainer(TrainerBuilder builder) {
        this.model = builder.model;
        this.optimizer = builder.optimizer;
        this.lossFunction = builder.lossFunction;
        this.patience = builder.patience;
        this.earlyStopping = builder.earlyStopping;
        this.verbose = builder.verbose;
        this.epochs = builder.epochs;
        this.evalEvery = builder.evalEvery;
        this.batchSize = builder.batchSize;
        double trainRatio = builder.trainRatio;
        this.trainInputs = builder.trainInputs;
        this.trainTargets = builder.trainTargets;
        if (builder.testInputs != null && builder.testTargets != null) {
            this.testInputs = builder.testInputs;
            this.testTargets = builder.testTargets;
        } else {
            this.splitData(builder.trainInputs, builder.trainTargets, trainRatio);
        }
    }

    public void fit() {
        System.out.println("Training...");
        System.out.println("Epochs: " + epochs);
        System.out.println("Batch size: " + batchSize);
        INDArray predictions = null, gradLoss = null;

        Runtime runtime = Runtime.getRuntime();
        long usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory();
        System.out.println("Used memory before training: " + usedMemoryBefore);

        long startTime = System.currentTimeMillis();

        for (int epoch = 0; epoch < epochs; epoch++) {
            currentIndex = 0;
            while (hasNextBatch()) {
                this.getNextBatch();

//                System.out.println("1x_train [0]: " + trainInputs.getRow(0));
//                System.out.println("1y_train [0]: " + trainTargets.getRow(0));

                // Forward pass
                predictions = model.predict(batch[0]);

//                System.out.println("2x_train [0]: " + trainInputs.getRow(0));
//                System.out.println("2y_train [0]: " + trainTargets.getRow(0));

                // Backward pass
                gradLoss = lossFunction.backward(predictions, batch[1]);

//                System.out.println("3x_train [0]: " + trainInputs.getRow(0));
//                System.out.println("3y_train [0]: " + trainTargets.getRow(0));
                //System.out.println("gradLoss: " + gradLoss);
                model.backPropagation(gradLoss);

                optimizer.update();
            }

            if ((epoch+1) % evalEvery == 0) {
                //double trainLoss = lossFunction.forward(model.predict(trainInputs), trainTargets).getDouble(0);
                valLoss = lossFunction.forward(model.predict(testInputs.dup()), testTargets.dup()).getDouble(0);

                if (verbose) {
                    Util.printProgressBar(epoch, epochs);
                    System.out.print(" Epoch: " + epoch + " Val Loss: " + valLoss);
                }
                if (earlyStopping) {
                    if (earlyStopping()) {
                        System.out.println("\nEarly stopping at epoch: " + epoch + " Loss: " + valLoss);
                        epoch = epochs;
                    }
                }
            }
        }

        long endTime = System.currentTimeMillis();
        System.out.println("\nTraining time: " + (endTime - startTime) / 1000.0 + " seconds");

        long usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory();
        System.out.println("Used memory after training: " + usedMemoryAfter);
        System.out.println("Memory used for training: " + (usedMemoryAfter - usedMemoryBefore));
    }

    public void evaluate() {
        double testLoss = lossFunction.forward(model.predict(testInputs), testTargets).getDouble(0) + threshold;
        System.out.println("Test Loss: " + testLoss);
    }

    private boolean earlyStopping() {
        if (valLoss < bestLoss - threshold) {
            wait = 0;
            bestLoss = valLoss;
        } else if ((valLoss > bestLoss + threshold) || (valLoss == bestLoss) || (valLoss < threshold)) {
            wait++;
        }

        return (valLoss <= threshold) || (wait >= patience);
    }

    private boolean hasNextBatch() {
        return currentIndex < trainInputs.size(0);
    }

    private void getNextBatch() {
        int end = (int) Math.min(currentIndex + batchSize, trainInputs.size(0));
        batch[0] = trainInputs.get(NDArrayIndex.interval(currentIndex, end));
        batch[1] = trainTargets.get(NDArrayIndex.interval(currentIndex, end));
        currentIndex += batchSize;
    }

    public void splitData(INDArray inputs, INDArray targets, double trainRatio) {
        INDArray[][] trainTestSplit = Util.trainTestSplit(inputs, targets, trainRatio);
        this.trainInputs = trainTestSplit[0][0];
        this.trainTargets = trainTestSplit[0][1];
        this.testInputs = trainTestSplit[1][0];
        this.testTargets = trainTestSplit[1][1];
    }

    public void printDataInfo() {
        System.out.println("Train inputs shape: " + trainInputs.shapeInfoToString());
        System.out.println("Train targets shape: " + trainTargets.shapeInfoToString());
        System.out.println("Test inputs shape: " + testInputs.shapeInfoToString());
        System.out.println("Test targets shape: " + testTargets.shapeInfoToString());
    }

    public INDArray getTrainInputs() {
        return this.trainInputs;
    }

    public INDArray getTrainTargets() {
        return this.trainTargets;
    }

    public INDArray getTestInputs() {
        return this.testInputs;
    }

    public INDArray getTestTargets() {
        return this.testTargets;
    }

}
