package br.edu.unifei.ecot12.deeplearning4java.game.controller;

import br.edu.unifei.ecot12.deeplearning4java.game.model.PredictionResult;
import br.edu.unifei.ecot12.deeplearning4java.game.viewmodel.GameViewModel;
import br.edu.unifei.ecot12.deeplearning4java.game.viewmodel.ViewModelManager;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.geometry.Rectangle2D;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.transform.Transform;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class GameController implements Initializable {

    private GameViewModel viewModel;
    public Label categoryLabel;
    public Label timeLabel;
    public Button eraseButton;
    public Button nextButton;
    public Button closeButton;
    public ListView listPredictions;
    @FXML
    private Canvas canvas;
    private GraphicsContext gc;


    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        System.out.println("GameController initialized");

        gc = canvas.getGraphicsContext2D();
        // Set canvas to gray scale
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(4);

        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, event -> {
            gc.beginPath();
            gc.moveTo(event.getX(), event.getY());
            gc.stroke();
        });

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, event -> {
            gc.lineTo(event.getX(), event.getY());
            gc.stroke();
        });

        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED, event -> {
            try {
                SnapshotParameters params = new SnapshotParameters();
                params.setFill(Color.WHITE);
                handleDrawing(canvas.snapshot(params, null));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            gc.closePath();
        });
    }

    public void handleDrawing(WritableImage drawing) throws IOException {
        viewModel.sendDrawing(drawing);
    }

    public void handleEraseButtonAction(ActionEvent actionEvent) {
        gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
    }

    public void handleNextButtonAction(ActionEvent actionEvent) throws IOException {
        viewModel.nextRound();
        gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        Node source;
        if (actionEvent != null) {
            source = (Node) actionEvent.getSource();
        } else {
            // Use a default source if actionEvent is null
            source = closeButton;
        }
        Stage stage = (Stage) source.getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/transition-view.fxml"));
        Parent root = loader.load();
        TransitionController controller = loader.getController();
        controller.setViewModel(ViewModelManager.getInstance().getTransitionViewModel(controller));
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }

    public void handleCloseButtonAction(ActionEvent actionEvent) throws IOException {
        Stage stage = (Stage) ((Node) actionEvent.getSource()).getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
        Parent root = loader.load();
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }

    public void updateTime(int time) {
        timeLabel.setText(String.format("%02d:%02d", time / 60, time % 60));
    }

    public void updateCategory(String category) {
        categoryLabel.setText("Draw: " + category);
    }

    public void updatePredictions(List<PredictionResult> predictions) {
        listPredictions.getItems().clear();
        for (PredictionResult prediction : predictions) {
            listPredictions.getItems().add(prediction.getCategory() + ": " + prediction.getProbability() + "%");
        }
    }

    public void setViewModel(GameViewModel gameViewModel) {
        this.viewModel = gameViewModel;
        viewModel.updateView();
    }

    public void endGame() throws IOException {
        // Send the image for database
        viewModel.sendDrawing(canvas.snapshot(null, null));
        // Return to menu equals close button
        Node source = closeButton;
        if (source.getScene() != null) {
            Stage stage = (Stage) source.getScene().getWindow();
            FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
            Parent root = loader.load();
            Scene scene = new Scene(root);
            stage.setScene(scene);
            stage.show();
        }
    }
}