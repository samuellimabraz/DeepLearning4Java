package br.edu.unifei.ecot12.deeplearning4java.game.viewmodel;

import br.edu.unifei.ecot12.deeplearning4java.game.controller.GameController;
import br.edu.unifei.ecot12.deeplearning4java.game.model.GameSession;
import br.edu.unifei.ecot12.deeplearning4java.game.model.PredictionResult;
import br.edu.unifei.ecot12.deeplearning4java.game.model.database.InputData;
import br.edu.unifei.ecot12.deeplearning4java.game.model.database.InputDataDao;
import br.edu.unifei.ecot12.deeplearning4java.game.model.database.InputDataDaoProxy;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.image.*;
import javafx.scene.paint.Color;
import javafx.scene.layout.Pane;
import javafx.embed.swing.SwingFXUtils;
import javafx.stage.Stage;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class GameViewModel extends ViewModel {
    private final GameController controller;

    public GameViewModel(GameController controller, GameSession session) {
        super(session);
        session.setViewModel(this);
        this.controller = controller;
    }


    public void nextRound() throws IOException {
        if (session.getCurrentRoundIndex() == 5) {
            this.endGame();
            return;
        }
        session.nextRound();
    }

    public void endGame() throws IOException {
        controller.endGame();
    }
    public void sendDrawing(WritableImage drawing) throws IOException {
        INDArray imgArray = convertImageToINDArray(drawing, 28, 28);
        displayImage(arrayToImage(imgArray));
        session.setDrawing(imgArray.divi(255.0));
        List<PredictionResult> predictions = session.predict();

        // Ordenar as predições em ordem decrescente de probabilidade
        predictions.sort((p1, p2) -> Double.compare(p2.getProbability(), p1.getProbability()));
        // Get 5 first predictions
        predictions = predictions.subList(0, 5);

        System.out.println("Predition result: " + predictions.get(0).getCategory());
        controller.updatePredictions(predictions);
        // Verificar se a predição está correta
        if (predictions.get(0).getCategory().equals(session.getCurrentRound().getCategory())) {
            System.out.println("Correct!");
            // Enviar a imagem para a simulação do banco de dados
            InputData inputData = new InputData(drawing, predictions.get(0).getCategory());
            InputDataDaoProxy daoProxy = new InputDataDaoProxy(new InputDataDao());
            daoProxy.save(inputData);

            try {
                Thread.sleep(1500); // Atraso de 2 segundos
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // Adicionar a lógica de transição de tela aqui
            Platform.runLater(() -> {
                try {
                    controller.handleNextButtonAction(null);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    public INDArray convertImageToINDArray(WritableImage writableImage, int width, int height) {
        // Convert the JavaFX image to a BufferedImage
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(writableImage, null);

        // Convert the BufferedImage to an OpenCV Mat
        int[] pixels = bufferedImage.getRGB(0, 0, bufferedImage.getWidth(), bufferedImage.getHeight(), null, 0, bufferedImage.getWidth());
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC4);
        byte[] data = new byte[pixels.length * (int)(mat.elemSize())];
        for (int i = 0; i < pixels.length; i++) {
            data[i*4] = (byte) ((pixels[i] >> 16) & 0xFF); // Red
            data[i*4 + 1] = (byte) ((pixels[i] >> 8) & 0xFF); // Green
            data[i*4 + 2] = (byte) ((pixels[i]) & 0xFF); // Blue
            data[i*4 + 3] = (byte) ((pixels[i] >> 24) & 0xFF); // Alpha
        }
        mat.put(0, 0, data);

        // Convert the resized image to grayscale
        Mat grayMat = new Mat();
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY);

        // Resize the image to 28x28
        Mat resizedMat = new Mat();
        Imgproc.resize(grayMat, resizedMat, new Size(width, height), 0, 0, Imgproc.INTER_AREA);;

        // Convert the grayscale image to an INDArray
        int totalBytes = (int) (resizedMat.total() * resizedMat.elemSize());
        byte[] buffer = new byte[totalBytes];
        resizedMat.get(0, 0, buffer);

        // Convert buffer bytes to unsigned integers
        double[] unsignedBuffer = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            unsignedBuffer[i] = ((int) buffer[i]) & 0xff;
        }

        System.out.println("unsignedBuffer min - max: " + Arrays.stream(unsignedBuffer).min().getAsDouble() + " - " + Arrays.stream(unsignedBuffer).max().getAsDouble());

        return Nd4j.createFromArray(unsignedBuffer).castTo(DataType.DOUBLE).reshape(1, (long) width * height);
    }

    private WritableImage arrayToImage(INDArray array) {
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

    public void updateTimer(int time) {
        if (time == 0) {
            System.out.println("Time is up!");
            Platform.runLater(() -> {
                try {
                    controller.handleNextButtonAction(null);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        } else {
            Platform.runLater(() -> controller.updateTime(time));
        }
    }

    @Override
    public void updateView() {
        controller.updateCategory(session.getCurrentRound().getCategory());
    }

    public void displayImage(WritableImage image) {
        // Criar uma nova janela
        Stage stage = new Stage();
        stage.setTitle("Converted Image");

        // Criar uma ImageView para exibir a imagem
        ImageView imageView = new ImageView(image);

        // Adicionar a ImageView a um novo Pane
        Pane pane = new Pane(imageView);

        // Criar uma nova Scene com o Pane
        Scene scene = new Scene(pane);

        // Definir a Scene para a Stage
        stage.setScene(scene);

        // Exibir a Stage
        stage.show();
    }

}
