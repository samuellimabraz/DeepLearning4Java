package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.qdraw;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class DataVisualizer extends Application {

    private INDArray xTrain;
    private INDArray yTrain;

    private INDArray xTest;

    private INDArray yTest;

    public void loadINDArrays() {
        String root = "src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw";
        xTrain = Nd4j.createFromNpyFile(new File(root + "/npy/train/x_train250.npy"));
        System.out.println(xTrain.shapeInfoToString());
        yTrain = Nd4j.createFromNpyFile(new File(root + "/npy/train/y_train250.npy"));
        System.out.println(yTrain.shapeInfoToString());
        xTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/x_test250.npy"));
        System.out.println(xTest.shapeInfoToString());
        yTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/y_test250.npy"));
        System.out.println(yTest.shapeInfoToString());

        System.out.println("xTrain [0]: " + xTrain.get(NDArrayIndex.point(0), NDArrayIndex.all()).reshape(28, 28));
        System.out.println("yTrain [0]: " + yTrain.getDouble(0));

    }

    @Override
    public void start(Stage primaryStage) {
        loadINDArrays();

        // Cria visualização com 10 imagens em cada linha, com 10 linhas
        VBox vbox = new VBox();
        for (int i = 0; i < 50; i++) {
            HBox row = new HBox();
            for (int j = 0; j < 40; j++) {
                int index = i * 40 + j;
                INDArray imgArray = xTrain.get(NDArrayIndex.point(index), NDArrayIndex.all()).reshape(28, 28);
                Image img = arrayToImage(imgArray);
                ImageView imageView = new ImageView(img);
                row.getChildren().add(imageView);
            }
            vbox.getChildren().add(row);
        }

        Scene scene = new Scene(vbox, 800, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private Image arrayToImage(INDArray array) {
        int width = (int) Math.sqrt(array.length());
        WritableImage img = new WritableImage(width, width);
        for (int i = 0; i < array.length(); i++) {
            double pixelValue = array.getDouble(i);
            int x = i % width;
            int y = i / width;
            Color color = Color.gray(pixelValue / 255.0);
            img.getPixelWriter().setColor(x, y, color);
        }
        return img;
    }

    public INDArray convertImageToINDArray(WritableImage writableImage) {
        // Redimensionar a imagem para 28x28 pixels

        ImageView imageView = new ImageView(writableImage);
        imageView.setFitWidth(28);
        imageView.setFitHeight(28);
        Image image = imageView.snapshot(null, null);

        // Criar um PixelReader para ler os pixels da imagem
        PixelReader pixelReader = image.getPixelReader();

        // Criar um array para armazenar os pixels da imagem
        double[] pixels = new double[28 * 28];

        // Converter a imagem para escala de cinza e armazenar os pixels no array
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                Color color = pixelReader.getColor(x, y);
                double gray = 0.299 * color.getRed() + 0.587 * color.getGreen() + 0.114 * color.getBlue();
                pixels[y * 28 + x] = gray;
            }
        }

        // Normalizar os valores dos pixels para o intervalo [0,1]
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] *= 255.0;
        }

        // Converter o array de pixels para um INDArray

        return Nd4j.create(pixels, new int[]{1, 784});
    }

    public static void main(String[] args) {
        launch(args);
    }
}

