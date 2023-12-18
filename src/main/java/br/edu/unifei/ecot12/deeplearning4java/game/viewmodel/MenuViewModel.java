package br.edu.unifei.ecot12.deeplearning4java.game.viewmodel;

import br.edu.unifei.ecot12.deeplearning4java.game.controller.MenuController;
import br.edu.unifei.ecot12.deeplearning4java.game.model.GameSession;

public class MenuViewModel extends ViewModel {
    private final MenuController controller;
    public MenuViewModel(MenuController menuController, GameSession session) {
        super(session);
        this.controller = menuController;
    }

    @Override
    public void updateView() {}
}
