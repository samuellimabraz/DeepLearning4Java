package br.edu.unifei.ecot12.deeplearning4java.game.viewmodel;

import br.edu.unifei.ecot12.deeplearning4java.game.model.GameSession;

public abstract class ViewModel {
    protected GameSession session;

    public ViewModel(GameSession session) {
        this.session = session;
    }

    public abstract void updateView();
}
