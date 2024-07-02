package br.deeplearning4java.game.viewmodel;

import br.deeplearning4java.game.model.GameSession;
import br.deeplearning4java.game.model.PredictionModel;
import br.deeplearning4java.game.controller.GameController;
import br.deeplearning4java.game.controller.TransitionController;
import br.deeplearning4java.game.model.database.PersistenceManager;

import javax.persistence.EntityManager;

public class ViewModelManager {
    private static ViewModelManager instance = new ViewModelManager();
    private GameSession currentSession;
    private EntityManager entityManager;

    private ViewModelManager() {}

    public static ViewModelManager getInstance() {
        return instance;
    }

    public void startNewSession(PredictionModel model, EntityManager em) {
        currentSession = new GameSession(model);
        entityManager = em;
        if (em != null) {
            System.out.println("Database Connected created");
            PersistenceManager.persist(em, currentSession.getModel());
            PersistenceManager.persist(em, currentSession);
        }
        currentSession.initRounds();
        System.out.println("New Session started");
        System.out.println("Model: " + currentSession.getModel().getClass().getSimpleName());
    }

    public GameSession getCurrentSession() {
        return currentSession;
    }

    public GameViewModel getGameViewModel(GameController controller) {
        return new GameViewModel(controller, currentSession, entityManager);
    }

    public TransitionViewModel getTransitionViewModel(TransitionController controller) {
        return new TransitionViewModel(controller, currentSession);
    }
}
