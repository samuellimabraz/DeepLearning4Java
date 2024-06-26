package br.deeplearning4java.game.controller;

import br.deeplearning4java.game.viewmodel.ViewModelManager;
import javafx.event.ActionEvent;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class MenuController {
    public void handleStartButtonAction(ActionEvent actionEvent) throws IOException {

        Stage stage = (Stage) ((Node) actionEvent.getSource()).getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(MenuController.class.getResource("fxml/transition-view.fxml"));
        Parent root = loader.load();
        TransitionController controller = loader.getController();
        controller.setViewModel(ViewModelManager.getInstance().getTransitionViewModel(controller));
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }
}
