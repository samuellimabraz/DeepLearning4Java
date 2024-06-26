package br.deeplearning4java.neuralnetwork.core.model;

import br.deeplearning4java.game.model.database.PersistenceManager;
import br.deeplearning4java.neuralnetwork.core.layers.Dense;
import br.deeplearning4java.neuralnetwork.core.layers.Layer;
import br.deeplearning4java.neuralnetwork.core.layers.TrainableLayer;
import br.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import org.junit.jupiter.api.Test;

import javax.persistence.EntityManager;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SaveLoadDataBase {

    @Test
    public void testSaveModel() throws Exception {
        // Carrega o modelo do arquivo
        String filePath = "src/test/java/br/deeplearning4java/neuralnetwork/core/model/sine_function.zip";
        NeuralNetwork loadedModel = null;
        try {
            loadedModel = NeuralNetwork.loadModel(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Salva o modelo no banco de dados
        EntityManager em = PersistenceManager.createEntityManager("neuralNetworkPU");
        PersistenceManager.persistAll(em, Arrays.asList(loadedModel.getLayers().toArray()));
        PersistenceManager.persist(em, loadedModel);

        // Carrega o modelo do banco de dados
        NeuralNetwork loadedModelFromDB = em.createQuery("SELECT n FROM NeuralNetwork n where n.name = :name", NeuralNetwork.class)
                .setParameter("name", loadedModel.getName())
                .getSingleResult();

        // Verifica se o modelo foi carregado corretamente
        assertEquals(loadedModel.getName(), loadedModelFromDB.getName());

        List<TrainableLayer> trainableLayers = loadedModel.getTrainableLayers();
        List<TrainableLayer> trainableLayersFromDB = loadedModelFromDB.getTrainableLayers();

        assertEquals(trainableLayers.size(), trainableLayersFromDB.size());
        for (int i = 0; i < trainableLayers.size(); i++) {
            TrainableLayer trainableLayer = trainableLayers.get(i);
            TrainableLayer trainableLayerFromDB = trainableLayersFromDB.get(i);
            assertEquals(trainableLayer.getName(), trainableLayerFromDB.getName());
            assertEquals(trainableLayer.getParams(), trainableLayerFromDB.getParams());
            assertEquals(trainableLayer.getGrads(), trainableLayerFromDB.getGrads());
        }

    }
}
