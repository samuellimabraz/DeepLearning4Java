package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.mnist;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MnistDataLoader {
    private final INDArray mnistTrainData;
    private final INDArray mnistTestData;

    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;

    public MnistDataLoader() throws IOException {
        File trainDataFile = new File("src/main/resources/data/mnist/train/mnist_train.bin");
        File testDataFile = new File("src/main/resources/data/mnist/test/mnist_test.bin");

        if (trainDataFile.exists() && testDataFile.exists()) {
            mnistTrainData = Nd4j.readBinary(trainDataFile);
            mnistTestData = Nd4j.readBinary(testDataFile);
        } else {
            mnistTrainData = loadCsv("src/main/resources/data/mnist/train/mnist_train.csv");
            mnistTestData = loadCsv("src/main/resources/data/mnist/test/mnist_test.csv");

            Nd4j.saveBinary(mnistTrainData, trainDataFile);
            Nd4j.saveBinary(mnistTestData, testDataFile);
        }
    }

    public static INDArray loadCsv(String csvFile) {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                double[] row = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Double.parseDouble(tokens[i]);
                }
                data.add(row);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("Loaded " + data.size() + " rows from " + csvFile);
        }

        return Nd4j.create(data.toArray(new double[0][]));
    }

    public INDArray getAllTrainImages() {
        return mnistTrainData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, mnistTrainData.columns()));
    }

    public INDArray getAllTestImages() {
        return mnistTestData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, mnistTestData.columns()));
    }

    public INDArray getAllTrainLabels() {
        return mnistTrainData.get(NDArrayIndex.all(), NDArrayIndex.point(0)).reshape(mnistTrainData.rows(), 1);
    }

    public INDArray getAllTestLabels() {
        return mnistTestData.get(NDArrayIndex.all(), NDArrayIndex.point(0)).reshape(mnistTestData.rows(), 1);
    }

    public INDArray getTrainImage(int index) {
        return mnistTrainData.getRow(index).get(NDArrayIndex.interval(1, mnistTrainData.columns()));
    }

    public INDArray getTestImage(int index) {
        return mnistTestData.getRow(index).get(NDArrayIndex.interval(1, mnistTestData.columns()));
    }

    public int getTrainLabel(int index) {
        return (int) mnistTrainData.getRow(index).getDouble(0);
    }

    public int getTestLabel(int index) {
        return (int) mnistTestData.getRow(index).getDouble(0);
    }

    public INDArray getMnistTrainData() {
        return mnistTrainData;
    }

    public INDArray getMnistTestData() {
        return mnistTestData;
    }


    public static void main(String[] args) {
        INDArray x_train = loadCsv("D:\\IdeaProjects\\deeplearning4java\\src\\main\\resources\\mnist_train.csv");
        INDArray x_test = loadCsv("D:\\IdeaProjects\\deeplearning4java\\src\\main\\resources\\mnist_test.csv");

        System.out.println("x_train.shape: " + x_train.shapeInfoToString());
        System.out.println("x_test.shape: " + x_test.shapeInfoToString());
    }
}
