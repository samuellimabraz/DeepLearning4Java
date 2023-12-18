package br.edu.unifei.ecot12.deeplearning4java.game.viewmodel;

import br.edu.unifei.ecot12.deeplearning4java.game.controller.GameController;
import br.edu.unifei.ecot12.deeplearning4java.game.controller.TransitionController;
import br.edu.unifei.ecot12.deeplearning4java.game.model.GameSession;
import br.edu.unifei.ecot12.deeplearning4java.game.model.PredictionModel;

public class ViewModelManager {
    private static ViewModelManager instance = new ViewModelManager();
    private GameSession currentSession;

    private ViewModelManager() {}

    public static ViewModelManager getInstance() {
        return instance;
    }

    public void startNewSession(PredictionModel model) {
        currentSession = new GameSession(model);
        System.out.println("New session started");
        System.out.println("Model: " + currentSession.getModel().getClass().getSimpleName());
    }

    public GameSession getCurrentSession() {
        return currentSession;
    }

    public GameViewModel getGameViewModel(GameController controller) {
        return new GameViewModel(controller, currentSession);
    }

    public TransitionViewModel getTransitionViewModel(TransitionController controller) {
        return new TransitionViewModel(controller, currentSession);
    }
}
