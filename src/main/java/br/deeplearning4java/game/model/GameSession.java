package br.deeplearning4java.game.model;

import br.deeplearning4java.game.model.database.PersistenceManager;
import br.deeplearning4java.game.viewmodel.GameViewModel;

import javax.persistence.*;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

@Entity
public class GameSession implements RoundListener {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Transient
    private GameViewModel viewModel;

    @OneToMany(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "game_session_id")
    private final List<Round> rounds = new ArrayList<>(4);

    private int currentRound;

    @OneToOne
    private PredictionModel model;

    private LocalDate date;
    private String country;

    public GameSession(PredictionModel model) {
        this.model = model;
        currentRound = 0;
        // Init 6 Rounds with random categories
        List<String> categories = new ArrayList<>(model.getCategories());
        for (int i = 0; i < 4; i++) {
            // Add random category of model categories
            String category = categories.get((int) (Math.random() * categories.size()));
            categories.remove(category);
            rounds.add(new Round(category, 20));
            rounds.get(i).setListener(this);
        }

        this.date = LocalDate.now();
        this.country = Locale.getDefault().getCountry();

        // Print the country and date information
        System.out.println("Game started in country: " + country + " at " + date);
    }

    public GameSession() {
    }

    public void start() {
        // Start first round
        rounds.get(currentRound).start();
    }

    public boolean nextRound() {
        cancelTimer();
        currentRound++;
        return currentRound != rounds.size();
    }

    public void endGame() {
        cancelTimer();
        currentRound = 0;
    }

    public void cancelTimer() {
        try {
            getCurrentRound().getTimer().cancel();
        } catch (Exception e) {
            //System.out.println("No timer to cancel");
        }
    }

    public Round getCurrentRound() {
        return rounds.get(currentRound);
    }

    public int getCurrentRoundIndex() {
        return currentRound;
    }

    public PredictionModel getModel() {
        return model;
    }

    public List<Round> getRounds() {
        return rounds;
    }

    public void setDrawing(Draw drawing) {
        this.rounds.get(currentRound).setDrawing(drawing);
    }

    public Draw getDrawing() {
        return rounds.get(currentRound).getDrawing();
    }

    public List<PredictionResult> predict() {
        return model.predict(rounds.get(currentRound).getDrawing().getData());
    }

    public void setViewModel(GameViewModel viewModel) {
        this.viewModel = viewModel;
    }

    public LocalDate getDate() {
        return date;
    }

    public String getCountry() {
        return country;
    }

    @Override
    public void onTimeUpdated(int time) {
        viewModel.updateTimer(time);
    }
}
