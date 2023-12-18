package br.edu.unifei.ecot12.deeplearning4java.game.model;

public class PredictionResult {
    private final String category;
    private final double probability;

    public PredictionResult(String category, double probability) {
        this.category = category;
        this.probability = probability;
    }

    public String getCategory() {
        return category;
    }

    public double getProbability() {
        return probability;
    }
}
