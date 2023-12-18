package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.mnist;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.image.*;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public class MnistPlotter extends Application {
    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;
    private static final int SCALE = 10;
    private int currentImageIndex = 0;

    @Override
    public void start(Stage primaryStage) throws IOException {
        primaryStage.setTitle("MNIST Image");

        MnistDataLoader mnistDataLoader = new MnistDataLoader();

        // Obter a primeira imagem
        INDArray imageArray = mnistDataLoader.getTrainImage(currentImageIndex);
        int label = mnistDataLoader.getTrainLabel(currentImageIndex);

        // Converter a imagem para um formato que o JavaFX pode usar
        WritableImage image = arrayToImage(imageArray);

        // Exibir a imagem
        ImageView imageView = new ImageView(image);
        imageView.setFitWidth(WIDTH * SCALE);  // Ajustar a largura da imagem
        imageView.setFitHeight(HEIGHT * SCALE);  // Ajustar a altura da imagem
        Label labelView = new Label("Label: " + label);
        HBox hbox = new HBox(imageView, labelView);
        Scene scene = new Scene(hbox, WIDTH * SCALE, HEIGHT * SCALE);
        primaryStage.setScene(scene);
        primaryStage.show();

        // Adicionar um manipulador de clique para alternar as imagens
        scene.setOnMouseClicked(event -> {
            currentImageIndex = (currentImageIndex + 1) % mnistDataLoader.getMnistTrainData().rows();
            INDArray nextImageArray = mnistDataLoader.getTrainImage(currentImageIndex);
            int nextLabel = mnistDataLoader.getTrainLabel(currentImageIndex);
            WritableImage nextImage = arrayToImage(nextImageArray);
            imageView.setImage(nextImage);
            labelView.setText("Label: " + nextLabel);
        });
    }

    public WritableImage arrayToImage(INDArray imageArray) {
        WritableImage image = new WritableImage(WIDTH, HEIGHT);
        PixelWriter pixelWriter = image.getPixelWriter();
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                double colorValue = imageArray.getDouble(y * WIDTH + x);
                Color color = Color.gray(colorValue / 255.0);
                pixelWriter.setColor(x, y, color);
            }
        }
        return image;
    }


    public static void main(String[] args) {
        launch(args);
    }
}

