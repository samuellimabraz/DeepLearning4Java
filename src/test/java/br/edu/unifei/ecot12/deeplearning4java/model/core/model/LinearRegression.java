package br.edu.unifei.ecot12.deeplearning4java.model.core.model;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.losses.MeanSquaredError;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.MAE;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.metrics.RMSE;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.ModelBuilder;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.optimizers.*;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.processing.DataProcessor;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.data.processing.StandardScaler;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.layers.Dense;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models.NeuralNetwork;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train.Trainer;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.train.TrainerBuilder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class LinearRegression {
    public static void main(String[] args) {
        // Criação dos dados de treinamento
        int numSamples = 300;
        INDArray x = Nd4j.linspace(1, numSamples, numSamples).reshape(numSamples, 1).castTo(DataType.DOUBLE);
        INDArray y = x.mul(2).add(40).add(Nd4j.randn(numSamples, 1).mul(7)).castTo(DataType.DOUBLE);

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        x = scaler.fitTransform(x);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(1, Activation.create("linear"), "xavier"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new SGDNesterov(0.02, 0.9);
        Trainer trainer = new TrainerBuilder(model, x, y, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(5000)
                .setBatchSize(32)
                .setTrainRatio(0.8)
                .setEarlyStopping(true)
                .setPatience(50)
                .setEvalEvery(10)
                .build();
        trainer.fit();

        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);

        // Métricas
        MAE mae = new MAE();
        RMSE rmse = new RMSE();
        System.out.println("MAE: " + mae.evaluate(y_test, predict));
        System.out.println("RMSE: " + rmse.evaluate(y_test, predict));


        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_test, predict, "Regressão Linear");

        // Salva o modelo
        try {
            model.saveModel("src/test/java/br/edu/unifei/ecot12/deeplearning4java/model/core/model/linear_regression.bin");
            System.out.println("Modelo salvo com sucesso!");
        } catch (Exception e) {
            e.printStackTrace();
        }

        INDArray params = model.getTrainableLayers().get(0).getParams();
        INDArray weights = params.get(NDArrayIndex.all(), NDArrayIndex.interval(0, params.columns() - 1));
        INDArray bias = params.getColumn(params.columns() - 1);
        System.out.println("Weights: " + weights);
        System.out.println("Bias: " + bias);
    }
}

