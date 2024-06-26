package br.deeplearning4java.neuralnetwork.core.models;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import br.deeplearning4java.neuralnetwork.core.layers.LayerLoader;
import br.deeplearning4java.neuralnetwork.core.layers.TrainableLayer;
import br.deeplearning4java.neuralnetwork.core.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.persistence.*;

@Entity
public class NeuralNetwork implements Iterable<TrainableLayer> {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Long id;

    public String name = "neural_network_" + UUID.randomUUID().toString();

    @OneToMany(orphanRemoval = true)
    private List<Layer> layers = new ArrayList<>();
    @Transient
    private List<TrainableLayer> trainableLayers = new ArrayList<>();

    @Transient
    private INDArray output = null;

    public NeuralNetwork(ModelBuilder modelBuilder) {
        this.setLayers(modelBuilder.layers);
        this.trainableLayers = layers.stream()
                .filter(layer -> layer instanceof TrainableLayer)
                .map(layer -> (TrainableLayer) layer)
                .collect(Collectors.toList());
    }

    public NeuralNetwork() {
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    @Override
    public Iterator<TrainableLayer> iterator() {
        return trainableLayers.iterator();
    }

    public INDArray predict(INDArray x) {
        output = x;
        for (Layer layer : layers)
            output = layer.forward(output);
        return output;
    }
    public void backPropagation(INDArray gradout) {
        for (int i = layers.size() - 1; i >= 0; i--)
            gradout = layers.get(i).backward(gradout);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public List<TrainableLayer> getTrainableLayers() {
        return trainableLayers;
    }

    private void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    public void setTrainable(boolean trainable) {
        trainableLayers.forEach(layer -> layer.setTrainable(trainable));
    }

    public void setInference(boolean inference) {
        layers.forEach(layer -> layer.setInference(inference));
    }

    public void saveModel(String filePath) throws IOException {
        DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(filePath)));

        // Salva o número de camadas
        dos.writeInt(layers.size());

        // Salva cada camada
        layers.forEach(layer -> {
            try {
                layer.save(dos);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });

        dos.close();

        System.out.println("Model saved successfully");
    }

    public static NeuralNetwork loadModel(String filePath) throws Exception {
        DataInputStream dis = new DataInputStream(new FileInputStream(filePath));

        // Carrega o número de camadas
        int numLayers = dis.readInt();

        // Carrega cada camada
        ModelBuilder modelBuilder = new ModelBuilder();
        for (int i = 0; i < numLayers; i++) {
            Layer<?> layer = LayerLoader.load(dis);
            modelBuilder.add(layer);
        }

        dis.close();

        System.out.println("Model loaded successfully56");

        return modelBuilder.build();
    }
}