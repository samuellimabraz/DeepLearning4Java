package br.deeplearning4java.neuralnetwork.database;

import br.deeplearning4java.neuralnetwork.core.layers.Conv2D;
import br.deeplearning4java.neuralnetwork.core.layers.Layer;
import br.deeplearning4java.neuralnetwork.core.layers.ZeroPadding2D;
import br.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import com.mongodb.client.MongoClients;
import dev.morphia.Datastore;
import dev.morphia.Morphia;
import dev.morphia.query.Query;
import dev.morphia.query.filters.Filter;
import dev.morphia.query.filters.Filters;

import java.util.List;

public class NeuralNetworkService {
    private final static String DATABASE = "deeplearning4java";
    private final static String MONGODB_URI = "mongodb+srv://samuellimabraz:hibana22@cluster0.bo7cqjk.mongodb.net/?appName=Cluster0";
    private final Datastore datastore;

    public NeuralNetworkService() {
        this.datastore = Morphia.createDatastore(MongoClients.create(MONGODB_URI));
    }

    public void saveModel(NeuralNetwork model) {
        List<Layer> layers = model.getLayers();
        for (Layer<?> layer : layers) {
            layer.save(datastore);
        }
        // Agora salva a neural network
        datastore.save(model);
    }

    public NeuralNetwork loadModel(String modelName) {
        Filter filter = Filters.eq("name", modelName);
        Query<NeuralNetwork> query = datastore.find(NeuralNetwork.class).filter(filter);
        return query.first();
    }

    public List<NeuralNetwork> getAllModels() {
        return datastore.find(NeuralNetwork.class).iterator().toList();
    }
}