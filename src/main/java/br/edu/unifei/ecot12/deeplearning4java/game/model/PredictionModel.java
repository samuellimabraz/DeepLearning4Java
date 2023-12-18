package br.edu.unifei.ecot12.deeplearning4java.game.model;

import javafx.scene.image.WritableImage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public abstract class PredictionModel {
    protected List<String> categories;
    public abstract List<PredictionResult> predict(WritableImage drawing);

    public abstract List<PredictionResult> predict(INDArray drawing);

    public abstract List<String> getCategories();

}
