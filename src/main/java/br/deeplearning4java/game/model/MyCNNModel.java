package br.deeplearning4java.game.model;

import br.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.deeplearning4java.neuralnetwork.core.activation.IActivation;
import br.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import br.deeplearning4java.neuralnetwork.data.Util;
import br.deeplearning4java.neuralnetwork.database.NeuralNetworkService;
import com.mongodb.client.MongoClients;
import dev.morphia.Datastore;
import dev.morphia.Morphia;
import dev.morphia.query.Query;
import dev.morphia.query.filters.Filter;
import dev.morphia.query.filters.Filters;
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
    private static NeuralNetwork model = null;
    @Transient
    private final IActivation softmax= Activation.create("softmax");

    public MyCNNModel() {
        categories = Arrays.asList(
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

        databaseType = "MongoDB";
    }

    @Override
    public void loadModel() {
        try {
            System.out.println("Loading model from MongoDB...");
            NeuralNetworkService service = new NeuralNetworkService();
            model = service.loadModel("quickdraw-cnn");
            model.setInference(true);
            modelName = MyCNNModel.model.name;
            modelLoaded = true;
            System.out.println("Model loaded successfully!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private INDArray preprocess(byte[] data) throws IOException {
        INDArray drawing = Util.bytesToINDArray(data, 28, 28);
        return drawing.divi(255.0).reshape(-1, 28, 28, 1);
    }

    @Override
    public List<PredictionResult> predict(byte[] data) {
        if (!modelLoaded) {
            throw new IllegalStateException("Model not loaded! Use loadModel() first.");
        }
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
