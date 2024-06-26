package br.deeplearning4java.game.controller;

import br.deeplearning4java.game.model.MyCNNModel;
import br.deeplearning4java.game.model.PredictionModel;
import br.deeplearning4java.game.model.database.PersistenceManager;
import br.deeplearning4java.game.viewmodel.ViewModelManager;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import nu.pattern.OpenCV;

import javax.persistence.EntityManager;
import java.io.IOException;

public class App extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        OpenCV.loadLocally();

        EntityManager gameEntityManager = PersistenceManager.createEntityManager("quickdrawPU");

        PredictionModel model = new MyCNNModel();

        ViewModelManager.getInstance().startNewSession(model, gameEntityManager);

        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 770, 620);
        stage.setTitle("Quick Draw!");
        stage.setScene(scene);
        stage.show();

        stage.setOnCloseRequest(e -> {
            PersistenceManager.persist(gameEntityManager, ViewModelManager.getInstance().getCurrentSession());
            gameEntityManager.close();
        });
    }

    public static void main(String[] args) {
        launch(args);
    }
}
