package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.qdraw;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.Dense;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.losses.CategoricalCrossEntropy;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.Accuracy;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.F1Score;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.Precision;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.Recall;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.ModelBuilder;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.optimizers.Optimizer;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.optimizers.RMSProp;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train.Trainer;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train.TrainerBuilder;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class QuickDrawNN {

    public static List<String> CLASS_NAMES = Arrays.asList(
            "cloud", "tedyy-bear", "basketball", "umbrella", "t-shirt", "baseball%20bat", "vase", "clock", "ladder", "tree"
    );

    public static void main(String[] args) throws Exception {
        testLoadandPredict();
    }

    public static void train() throws Exception {
        String root = "src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw";
        int numClasses = CLASS_NAMES.size();
        int numSamplesPerClass = 250;
        INDArray xTrain = Nd4j.createFromNpyFile(new File(root + "/npy/train/x_train" + numSamplesPerClass + ".npy"));
        System.out.println(xTrain.shapeInfoToString());
        INDArray yTrain = Nd4j.createFromNpyFile(new File(root + "/npy/train/y_train" + numSamplesPerClass + ".npy")).reshape(-1, 1);
        System.out.println(yTrain.shapeInfoToString());
        INDArray xTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/x_test" + numSamplesPerClass + ".npy"));
        System.out.println(xTest.shapeInfoToString());
        INDArray yTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/y_test" + numSamplesPerClass + ".npy")).reshape(-1, 1);
        System.out.println(yTest.shapeInfoToString());

        // Reshape
        xTrain = xTrain.reshape(-1, 784);
        xTest = xTest.reshape(-1, 784);
        System.out.println("xTrain shape: " + Arrays.toString(xTrain.shape()));
        System.out.println("xTest shape: " + Arrays.toString(xTest.shape()));


        // Normalization
        System.out.println("xTrain min - max: " + xTrain.minNumber() + " - " + xTrain.maxNumber());
        xTrain = xTrain.divi(255);
        xTest = xTest.divi(255);
        System.out.println("xTrain min - max: " + xTrain.minNumber() + " - " + xTrain.maxNumber());


        // One-hot encoding
        System.out.println("yTrain min - max: " + yTrain.minNumber() + " - " + yTrain.maxNumber());
        yTrain = Util.oneHotEncode(yTrain, 10).reshape(-1, 10);
        yTest = Util.oneHotEncode(yTest, 10).reshape(-1, 10);
        System.out.println("yTrain min - max: " + yTrain.minNumber() + " - " + yTrain.maxNumber());
        System.out.println("yTrain [0]: " + yTrain.getRow(0));

        // Build the model
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(89,  Activation.create("relu"), "xavier"))
                //.add(new Dense(36,  Activation.create("relu"), "xavier"))
                //.add(new Dense(36,  Activation.create("relu"), "xavier"))
                .add(new Dense(10,  Activation.create("softmax"), "xavier"))
                .build();

        Optimizer optimizer = new RMSProp(0.0002);
        Trainer trainer = new TrainerBuilder(model, xTrain, yTrain, xTest, yTest, new CategoricalCrossEntropy())
                .setOptimizer(optimizer)
                .setBatchSize(45)
                .setEpochs(80)
                .setEvalEvery(10)
                .setEarlyStopping(true)
                .setPatience(3)
                .build();

        trainer.printDataInfo();
        trainer.fit();

        // Test Predictions
        System.out.println("y_train[0] class: " + CLASS_NAMES.get(yTrain.getRow(0).argMax().getInt(0)));
        System.out.println("model.predict(x_train[0]): " + CLASS_NAMES.get(model.predict(xTrain.get(NDArrayIndex.point(0), NDArrayIndex.all()).reshape(1, 784)).argMax().getInt(0)));
        System.out.println("y_test[0] class: " + CLASS_NAMES.get(yTest.getRow(0).argMax().getInt(0)));
        System.out.println("model.predict(x_test[0]): " + CLASS_NAMES.get(model.predict(xTest.get(NDArrayIndex.point(0), NDArrayIndex.all()).reshape(1, 784)).argMax().getInt(0)));

       try {
           model.saveModel("src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw/model/quickdraw_modelv1.zip");
           System.out.println("Model saved");
       } catch (Exception e) {
           e.printStackTrace();
       }

        printMetrics(model.predict(xTrain), yTrain);
        printMetrics(model.predict(xTest), yTest);

    }

    public static void printMetrics(INDArray predict, INDArray y_test) {
        INDArray predictClass = predict.argMax(1);
        INDArray y_testClass = y_test.argMax(1);

        // Comparação entre os valores reais e os valores previstos
        INDArray comp = predictClass.eq(y_testClass).castTo(Nd4j.defaultFloatingPointType());
        System.out.println("\nPredict vs y_test: ");
        System.out.println("True: " + comp.sumNumber().intValue());
        System.out.println("False: " + (y_testClass.length() - comp.sumNumber().intValue()));

        INDArray predictOneHot = Util.oneHotEncode(predictClass.reshape(predictClass.length(), 1), 13);

        // Metricas
        Accuracy accuracy = new Accuracy();
        double acc = accuracy.evaluate(y_testClass, predictClass);
        System.out.println("Accuracy: " + acc);
        Precision precision = new Precision();
        double prec = precision.evaluate(y_test, predictOneHot);
        System.out.println("Precision: " + prec);
        Recall recall = new Recall();
        double rec = recall.evaluate(y_test, predictOneHot);
        System.out.println("Recall: " + rec);
        F1Score f1Score = new F1Score();
        double f1 = f1Score.evaluate(y_test, predictOneHot);
        System.out.println("F1 Score: " + f1);
    }

    public static void testLoadandPredict() throws Exception {
        NeuralNetwork model = NeuralNetwork.loadModel("src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw/model/quickdraw_modelv1.zip");
        System.out.println("Model loaded");
        String root = "src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw";
        int numClasses = CLASS_NAMES.size();
        int numSamplesPerClass = 250;
        INDArray xTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/x_test" + numSamplesPerClass + ".npy"));
        System.out.println(xTest.shapeInfoToString());
        INDArray yTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/y_test" + numSamplesPerClass + ".npy")).reshape(-1, 1);
        System.out.println(yTest.shapeInfoToString());

        // Reshape
        xTest = xTest.reshape(-1, 784);
        System.out.println("xTest shape: " + Arrays.toString(xTest.shape()));
        xTest = xTest.divi(255);

        yTest = Util.oneHotEncode(yTest, 10).reshape(-1, 10);
        System.out.println("yTest min - max: " + yTest.minNumber() + " - " + yTest.maxNumber());

        System.out.println("y_test[0] class: " + CLASS_NAMES.get(yTest.getRow(10).argMax().getInt(0)));
        System.out.println("model.predict(x_test[0]): " + CLASS_NAMES.get(model.predict(xTest.get(NDArrayIndex.point(10), NDArrayIndex.all()).reshape(1, 784)).argMax().getInt(0)));

    }
}
