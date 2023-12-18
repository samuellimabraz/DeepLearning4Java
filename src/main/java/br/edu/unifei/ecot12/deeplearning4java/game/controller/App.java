package br.edu.unifei.ecot12.deeplearning4java.game.controller;

import br.edu.unifei.ecot12.deeplearning4java.game.model.MultiLayerModel;
import br.edu.unifei.ecot12.deeplearning4java.game.model.PredictionModel;
import br.edu.unifei.ecot12.deeplearning4java.game.viewmodel.ViewModelManager;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class App extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        System.load("D:\\opencv\\build\\java\\x64\\opencv_java481.dll");
        PredictionModel model = new MultiLayerModel();
        ViewModelManager.getInstance().startNewSession(model);

        System.out.println(App.class);
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 770, 620);
        stage.setTitle("Quick Draw!");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
