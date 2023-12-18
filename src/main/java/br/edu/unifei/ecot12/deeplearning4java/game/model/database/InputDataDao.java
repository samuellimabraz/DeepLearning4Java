package br.edu.unifei.ecot12.deeplearning4java.game.model.database;

import javafx.geometry.Point2D;
import javafx.scene.image.WritableImage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class InputDataDao implements IInputDataDao {
    @Override
    public void save(InputData data) {
        INDArray image = imageToINDArray(data.getImage());
        INDArray points = pointsToINDArray(data.getPoints());
        // Salve os dados no banco de dados
    }

    @Override
    public InputData load(int id) {
        // Carregue os dados do banco de dados
        INDArray image = null;
        INDArray points = null;
        String category = null;
        return new InputData(indArrayToImage(image), category);
    }

    private INDArray imageToINDArray(WritableImage image) {
        // Implemente a conversão de WritableImage para INDArray
        return null;
    }

    private INDArray pointsToINDArray(List<Point2D> points) {
        // Implemente a conversão de List<Point> para INDArray
        return null;
    }

    private WritableImage indArrayToImage(INDArray array) {
        // Implemente a conversão de INDArray para WritableImage
        return null;
    }

    private List<Point2D> indArrayToPoints(INDArray array) {
        // Implemente a conversão de INDArray para List<Point2D>
        return null;
    }

}
