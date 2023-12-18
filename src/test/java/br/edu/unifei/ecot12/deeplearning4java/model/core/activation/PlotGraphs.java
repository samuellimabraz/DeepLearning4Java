package br.edu.unifei.ecot12.deeplearning4java.model.core.activation;

import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.ActivateEnum;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.activation.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.math.plot.Plot2DPanel;
import javax.swing.JFrame;

public class PlotGraphs {
    public static void createDataset(ActivateEnum type) {
        IActivation func = Activation.create(type);
        String name = func.getClass().getSimpleName();

        double[] xData = new double[201];
        double[] yData = new double[201];
        double[] yDataDerivative = new double[201];

        int i = 0;
        for (double x = -10; x <= 10; x += 0.1) {
            INDArray input = Nd4j.scalar(x);
            double y = func.forward(input).getDouble(0);
            double yDerivative = func.backward(input).getDouble(0);

            xData[i] = x;
            yData[i] = y;
            yDataDerivative[i] = yDerivative;
            i++;
        }

        plot(name, xData, yData, yDataDerivative);
    }

    public static void plot(String name, double[] xData, double[] yData, double[] yDataDerivative) {
        // Create panel with chart
        Plot2DPanel plot = new Plot2DPanel();
        plot.addLinePlot(name, xData, yData);
        plot.addLinePlot("Derivative of " + name, xData, yDataDerivative);
        //plot.addLinePlot("Sigmoid", xData, Transforms.sigmoid(Nd4j.create(xData), true).toDoubleVector());

        // Create frame and add panel
        JFrame frame = new JFrame();
        frame.setContentPane(plot);
        frame.setSize(800, 600);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        createDataset(ActivateEnum.TANH);
    }
}

