package br.edu.unifei.ecot12.deeplearning4java.game.viewmodel;

import br.edu.unifei.ecot12.deeplearning4java.game.controller.TransitionController;
import br.edu.unifei.ecot12.deeplearning4java.game.model.GameSession;

public class TransitionViewModel extends ViewModel {
    private final TransitionController controller;

    public TransitionViewModel(TransitionController controller, GameSession session) {
        super(session);
        this.controller = controller;
    }

    public void startRound() {
        session.start();
        updateView();
    }

    @Override
    public void updateView() {
        // Update the view based on the data in session
        // For example, you might want to update the current round and category
        controller.setRound(session.getCurrentRoundIndex() + 1);
        controller.setCategory(session.getCurrentRound().getCategory());
    }
}
