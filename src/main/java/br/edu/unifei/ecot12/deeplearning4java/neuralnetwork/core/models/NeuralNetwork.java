package br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.Layer;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.LayerLoader;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.TrainableLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NeuralNetwork implements Iterable<TrainableLayer> {
    private List<Layer> layers = new ArrayList<>();
    private List<TrainableLayer> trainableLayers = new ArrayList<>();
    private INDArray output = null;

    public NeuralNetwork(ModelBuilder modelBuilder) {
        this.setLayers(modelBuilder.layers);
        this.trainableLayers = layers.stream()
                .filter(layer -> layer instanceof TrainableLayer)
                .map(layer -> (TrainableLayer) layer)
                .collect(Collectors.toList());
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

        return modelBuilder.build();
    }
}