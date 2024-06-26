package br.deeplearning4java.game.model;

import br.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.deeplearning4java.neuralnetwork.core.activation.IActivation;
import br.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import br.deeplearning4java.neuralnetwork.data.Util;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.persistence.Entity;
import javax.persistence.Transient;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Entity
public class MyCNNModel extends PredictionModel {
    @Transient
    private static NeuralNetwork model;
    @Transient
    private final IActivation softmax= Activation.create("softmax");

    static {
        PredictionModel.categories = Arrays.asList(
                "ladder",
                "bucket",
                "t-shirt",
                "tree",
                "dumbbell",
                "clock",
                "square",
                "triangle",
                "hourglass",
                "candle"
        );

        try {
            MyCNNModel.model = NeuralNetwork.loadModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/quickdraw/model/quickdraw_model-cnn.zip");
            MyCNNModel.model.setInference(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public MyCNNModel() {
        super();
    }

    private INDArray preprocess(byte[] data) throws IOException {
        INDArray drawing = Util.bytesToINDArray(data, 28, 28);
        return drawing.divi(255.0).reshape(-1, 28, 28, 1);
    }

    @Override
    public List<PredictionResult> predict(byte[] data) {
        List<PredictionResult> results = new ArrayList<>();
        try {
            INDArray drawing = preprocess(data);
            INDArray percentages = this.softmax.forward(model.predict(drawing));
            System.out.println("Predictions: " + percentages);
            for (int i = 0; i < percentages.columns(); i++) {
                results.add(new PredictionResult(getCategories().get(i), percentages.getDouble(i) * 100.0));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return results;
    }
}
