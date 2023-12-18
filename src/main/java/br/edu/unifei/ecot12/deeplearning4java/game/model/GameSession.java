package br.edu.unifei.ecot12.deeplearning4java.game.model;

import br.edu.unifei.ecot12.deeplearning4java.game.viewmodel.GameViewModel;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class GameSession implements RoundListener {

    private GameViewModel viewModel;
    private List<Round> rounds = new ArrayList<>(6);
    private int currentRound = 0;

    private PredictionModel model;

    private INDArray drawing;

    public GameSession(PredictionModel model) {
        this.model = model;
        currentRound = 0;
        // Init 6 Rounds with random categories
        for (int i = 0; i < 6; i++) {
            // Add random category of model categories
            rounds.add(new Round(model.getCategories().get((int) (Math.random() * model.getCategories().size())), 20));
            rounds.get(i).setListener(this);
        }
    }

    public void start() {
        // Start first round
        rounds.get(currentRound).start();
    }

    public void nextRound() throws IOException {
        rounds.get(currentRound).getTimer().cancel();
        currentRound++;
    }

    public Round getCurrentRound() {
        return rounds.get(currentRound);
    }

    public int getCurrentRoundIndex() {
        return currentRound;
    }

    public void setModel(PredictionModel model) {
        this.model = model;
    }

    public PredictionModel getModel() {
        return model;
    }

    public List<Round> getRounds() {
        return rounds;
    }

    public void setRounds(List<Round> rounds) {
        this.rounds = rounds;
    }

    public void setDrawing(INDArray drawing) {
        this.drawing = drawing;
    }

    public List<PredictionResult> predict() {
        return model.predict(drawing);
    }

    public void setViewModel(GameViewModel viewModel) {
        this.viewModel = viewModel;
    }


    /**
     * @param time
     */
    @Override
    public void onTimeUpdated(int time) {
        viewModel.updateTimer(time);
    }
}
