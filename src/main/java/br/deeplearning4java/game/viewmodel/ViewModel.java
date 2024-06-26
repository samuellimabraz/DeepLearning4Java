package br.deeplearning4java.game.viewmodel;

import br.deeplearning4java.game.model.GameSession;

public abstract class ViewModel {
    protected GameSession session;

    public ViewModel(GameSession session) {
        this.session = session;
    }

    public abstract void updateView();
}
