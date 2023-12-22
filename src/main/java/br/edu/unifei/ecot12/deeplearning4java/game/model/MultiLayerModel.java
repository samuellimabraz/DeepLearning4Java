package br.edu.unifei.ecot12.deeplearning4java.game.model;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import javafx.scene.image.WritableImage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultiLayerModel extends PredictionModel {
    private NeuralNetwork model;
    public MultiLayerModel() {
        super();
        this.categories = Arrays.asList(
                "cloud", "tedyy bear", "basketball", "umbrella", "t-shirt", "baseball bat", "vase", "clock", "ladder", "tree");
        try {
            this.model = NeuralNetwork.loadModel("src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw/model/quickdraw_modelv1.zip");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public List<PredictionResult> predict(WritableImage drawing) {
        // Return randoms predictions
        List<PredictionResult> predictions = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            predictions.add(new PredictionResult(getCategories().get((int) (Math.random() * getCategories().size())), Math.random()));
        }
        return predictions;
    }

    @Override
    public List<PredictionResult> predict(INDArray drawing) {
        INDArray percentages = model.predict(drawing).mul(100);
        System.out.println("Predictions: " + percentages);
        List<PredictionResult> results = new ArrayList<>();
        for (int i = 0; i < percentages.columns(); i++) {
            results.add(new PredictionResult(getCategories().get(i), percentages.getDouble(i)));
        }
        return results;
    }

    @Override
    public List<String> getCategories() {
        return this.categories;
    }
}
