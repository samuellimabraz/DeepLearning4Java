package br.deeplearning4java.game.viewmodel;

import br.deeplearning4java.game.model.GameSession;
import br.deeplearning4java.game.controller.MenuController;

public class MenuViewModel extends ViewModel {
    private final MenuController controller;
    public MenuViewModel(MenuController menuController, GameSession session) {
        super(session);
        this.controller = menuController;
    }

    @Override
    public void updateView() {}
}
