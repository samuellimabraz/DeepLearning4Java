package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.mnist;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.Dense;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.losses.CategoricalCrossEntropy;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.Accuracy;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.F1Score;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.ModelBuilder;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.optimizers.*;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train.Trainer;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train.TrainerBuilder;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.processing.StandardScaler;
import org.nd4j.linalg.api.ndarray.INDArray;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.Util;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.Arrays;

public class MnistNN {
    public static void main(String[] args) throws IOException {
            MnistDataLoader mnistDataLoader = new MnistDataLoader();

            // Images
            INDArray x_train = mnistDataLoader.getAllTrainImages();
            INDArray x_test = mnistDataLoader.getAllTestImages();
            INDArray y_train = mnistDataLoader.getAllTrainLabels();
            INDArray y_test = mnistDataLoader.getAllTestLabels();

            // Pega uma fatia dos dados
            x_train = x_train.get(NDArrayIndex.interval(0, 5000));
            y_train = y_train.get(NDArrayIndex.interval(0, 5000));
            x_test = x_test.get(NDArrayIndex.interval(0, 200));
            y_test = y_test.get(NDArrayIndex.interval(0, 200));


            // Min-Max Scaling
            x_train = x_train.divi(255); // Normalization (0 - 1)
            x_test = x_test.divi(255); // Normalization (0 - 1)

            // Standardization
            StandardScaler standardScaler = new StandardScaler();
            standardScaler.fit(x_train);
            x_train = standardScaler.transform(x_train);
            x_test = standardScaler.transform(x_test);


            System.out.println("x_train min - max: " + x_train.minNumber() + " - " + x_train.maxNumber());
            System.out.println("x_test min - max: " + x_test.minNumber() + " - " + x_test.maxNumber());

            // One hot encoding
            y_train = Util.oneHotEncode(y_train, 10);
            y_test = Util.oneHotEncode(y_test, 10);

            System.out.println("y_test example: " + y_test.getRow(0));

            // Network
            NeuralNetwork model = new ModelBuilder()
                    .add(new Dense(89, Activation.create("tanh"), "xavier"))
                    .add(new Dense(10, Activation.create("softmax"), "xavier"))
                    .build();

            // Training
            Optimizer optimizer = new RMSProp(0.01, 0.9, 1e-8);
            Trainer trainer = new TrainerBuilder(model, x_train, y_train, x_test, y_test, new CategoricalCrossEntropy())
                    .setOptimizer(optimizer)
                    .setEpochs(200)
                    .setBatchSize(60)
                    .setEarlyStopping(false)
                    .setPatience(100)
                    .setEvalEvery(2)
                    .build();

            trainer.printDataInfo();
            trainer.fit();

            // Evaluation
            INDArray predictions = model.predict(x_test);
            //System.out.println("Predictions: " + predictions);
            System.out.println("Predictions shape: " + Arrays.toString(predictions.shape()));
            System.out.println("Predictions range: " + predictions.minNumber() + " - " + predictions.maxNumber());

            INDArray predictionsClasses = predictions.argMax(1);
            INDArray predictionsClassesOneHot = Util.oneHotEncode(predictionsClasses.reshape(predictionsClasses.length(), 1), 10);
            INDArray testLabelsClasses = y_test.argMax(1);


            Accuracy accuracy = new Accuracy();
            double acc = accuracy.evaluate(testLabelsClasses, predictionsClasses);
            System.out.println("Accuracy: " + acc);
            F1Score f1Score = new F1Score();
            double f1 = f1Score.evaluate(y_test, predictionsClassesOneHot);
            System.out.println("F1 Score: " + f1);

            try {
                model.saveModel("src/main/resources/data/mnist/model/mnist_model.bin");
                System.out.println("Modelo salvo com sucesso!");
            } catch (IOException e) {
                e.printStackTrace();
            }
    }
}
